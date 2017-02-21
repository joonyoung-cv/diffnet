local ffi = require 'ffi'
local task = torch.class( 'TaskManager' )
--------------------------------------------
-------- TASK-INDEPENDENT FUNCTIONS --------
--------------------------------------------
function task:__init(  )
	self.opt = {  }
	self.dbtr = {  }
	self.dbval = {  }
	self.inputStat = {  }
	self.numBatchTrain = 0
	self.numBatchVal = 0
	self.numQuery = 0
end
function task:setOption( arg )
	self.opt = self:parseOption( arg )
	self:setModelSpecificOption(  )
	assert( self.opt.numGpu )
	assert( self.opt.backend )
	assert( self.opt.numDonkey )
	assert( self.opt.data )
	assert( self.opt.numEpoch )
	assert( self.opt.epochSize )
	assert( self.opt.batchSize )
	assert( self.opt.learnRate )
	assert( self.opt.momentum )
	assert( self.opt.weightDecay )
	assert( self.opt.startFrom )
	assert( self.opt.dirRoot )
	assert( self.opt.pathDbTrain )
	assert( self.opt.pathDbVal )
	assert( self.opt.pathImStat )
	assert( self.opt.dirModel )
	assert( self.opt.pathModel )
	assert( self.opt.pathOptim )
	assert( self.opt.pathTrainLog )
	assert( self.opt.pathValLog )
	assert( self.opt.pathTestLog )
	paths.mkdir( self.opt.dirRoot )
	paths.mkdir( self.opt.dirModel )
	print( self.opt )
end
function task:getOption(  )
	return self.opt
end
function task:setDb(  )
	paths.dofile( string.format( '../db/%s.lua', self.opt.data ) )
	if paths.filep( self.opt.pathDbTrain ) then
		self:print( 'Load train db.' )
		self.dbtr = torch.load( self.opt.pathDbTrain )
		self:print( 'Done.' )
	else
		self:print( 'Create train db.' )
		self.dbtr = self:createDbTrain(  )
		torch.save( self.opt.pathDbTrain, self.dbtr )
		self:print( 'Done.' )
	end
	if paths.filep( self.opt.pathDbVal ) then
		self:print( 'Load val db.' )
		self.dbval = torch.load( self.opt.pathDbVal )
		self:print( 'Done.' )
	else
		self:print( 'Create val db.' )
		self.dbval = self:createDbVal(  )
		torch.save( self.opt.pathDbVal, self.dbval )
		self:print( 'Done.' )
	end
	self.numBatchTrain, self.numBatchVal = self:setNumBatch(  )
	self.numQuery = self:setNumQuery(  )
	assert( self.numBatchTrain > 0 )
	assert( self.numBatchVal > 0 )
	assert( self.numQuery > 0 )
end
function task:getNumBatch(  )
	return self.numBatchTrain, self.numBatchVal
end
function task:getNumQuery(  )
	return self.numQuery
end
function task:setInputStat(  )
	if self.opt.caffeInput then 
		self.opt.pathImStat = self.opt.pathImStat:match( '(.+).t7$' ) .. 'Caffe.t7' 
	end
	if paths.filep( self.opt.pathImStat ) then
		self:print( 'Load input data statistics.' )
		self.inputStat = torch.load( self.opt.pathImStat )
		self:print( 'Done.' )
	else
		self:print( 'Estimate input data statistics.' )
		self.inputStat = self:estimateInputStat(  )
		torch.save( self.opt.pathImStat, self.inputStat )
		self:print( 'Done.' )
	end
end
function task:getFunctionTrain(  )
	return
		function( x ) self:changeModelTrain( x ) end,
		function(  ) return self:getBatchTrain(  ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getFunctionVal(  )
	return
		function( x ) self:changeModelVal( x ) end,
		function( i ) return self:getBatchVal( i ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getFunctionTest(  )
	return
		function( x ) self:changeModelTest( x ) end,
		function( i ) return self:getQuery( i ) end,
		function( x ) return self:aggregateAnswers( x ) end,
		function( x, y ) return self:evaluate( x, y ) end
end
function task:getModel(  )
	local numEpoch = self.opt.numEpoch
	local pathModel = self.opt.pathModel
	local pathOptim = self.opt.pathOptim
	local numGpu = self.opt.numGpu
	local startFrom = self.opt.startFrom
	local backend = self.opt.backend
	local startEpoch = 1
	for e = 1, numEpoch do
		local modelPath = pathModel:format( e )
		local optimPath = pathOptim:format( e )
		if not paths.filep( modelPath ) then startEpoch = e break end 
		if e == numEpoch then self:print( 'All done.\n\n' ) os.exit(  ) end
	end
	local model, params, grads, optims
	if startEpoch == 1 and startFrom:len(  ) == 0 then
		self:print( 'Create model.' )
		model = self:defineModel(  )
		if backend == 'cudnn' then
			require 'cudnn'
			cudnn.convert( model, cudnn )
		end
		params, grads, optims = self:groupParams( model )
	elseif startEpoch == 1 and startFrom:len(  ) > 0 then
		self:print( 'Load user-defined model.' .. startFrom )
		model = loadDataParallel( startFrom, numGpu, backend )
		params, grads, optims = self:groupParams( model )
	elseif startEpoch > 1 then
		self:print( string.format( 'Load model from epoch %d.', startEpoch - 1 ) )
		model = loadDataParallel( pathModel:format( startEpoch - 1 ), numGpu, backend )
		params, grads, _ = self:groupParams( model )
		optims = torch.load( pathOptim:format( startEpoch - 1 ) )
	end
	self:print( 'Done.' )
	local criterion = self:defineCriterion(  )
	self:print( 'Model looks' )
	print( model )
	print(criterion)
	self:print( 'Convert model to cuda.' )
	model = model:cuda(  )
	criterion:cuda(  )
	self:print( 'Done.' )
	cutorch.setDevice( 1 )
	local modelSet = {  }
	modelSet.model = model
	modelSet.criterion = criterion
	modelSet.params = params
	modelSet.grads = grads
	modelSet.optims = optims
	-- Verification
	assert( #self.opt.learnRate == #modelSet.params )
	assert( #self.opt.learnRate == #modelSet.grads )
	assert( #self.opt.learnRate == #modelSet.optims )
	for g = 1, #modelSet.params do
		assert( modelSet.params[ g ]:numel(  ) == modelSet.grads[ g ]:numel(  ) )
	end
	return modelSet, startEpoch
end
function task:print( str )
	print( 'TASK MANAGER) ' .. str )
end
-----------------------------------------
-------- TASK-SPECIFIC FUNCTIONS --------
-----------------------------------------
function task:parseOption( arg )
	local cmd = torch.CmdLine(  )
	cmd:option( '-task', arg[ 2 ] )
	-- System.
	cmd:option( '-numGpu', 4, 'Number of GPUs.' )
	cmd:option( '-backend', 'cudnn', 'cudnn or nn.' )
	cmd:option( '-numDonkey', 16, 'Number of donkeys for data loading.' )
	-- Data.
	cmd:option( '-data', 'UCF101_OF_S1', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 256, 'Short side of initial resize.' )
	-- Model.
	cmd:option( '-numFlow', 10, 'Number of flow frames per chunk.' )
	cmd:option( '-dropout', 0.7, 'Dropout ratio.' )
	-- Train.
	cmd:option( '-numEpoch', 50, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 76, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 1280, 'Frame-level mini-batch size.' )
	cmd:option( '-learnRate', '1e-2,1e-2', 'Supports multi-lr for multi-module like "lr1,lr2,lr3".' )
	cmd:option( '-momentum', 0.9, 'Momentum.' )
	cmd:option( '-weightDecay', 5e-4, 'Weight decay.' )
	cmd:option( '-startFrom', '', 'Path to the initial model. Using it for LR decay is recommended.' )
	local opt = cmd:parse( arg or {  } )
	-- Set dst paths.
	local dirRoot = paths.concat( gpath.dataout, opt.data )
	local pathDbTrain = paths.concat( dirRoot, 'dbTrain.t7' )
	local pathDbVal = paths.concat( dirRoot, 'dbVal.t7' )
	local pathImStat = paths.concat( dirRoot, 'inputStat.t7' )
	local ignore = { numGpu=true, backend=true, numDonkey=true, data=true, numEpoch=true, startFrom=true }
	local dirModel = paths.concat( dirRoot, cmd:string( opt.task, opt, ignore ) )
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match( '(.+)/model_(%d+).t7' )
		dirModel = paths.concat( baseDir, cmd:string( 'model_' .. epoch, opt, ignore ) )
	end
	opt.dirRoot = dirRoot
	opt.pathDbTrain = pathDbTrain
	opt.pathDbVal = pathDbVal
	opt.pathImStat = pathImStat
	opt.dirModel = dirModel
	opt.pathModel = paths.concat( opt.dirModel, 'model_%03d.t7' )
	opt.pathOptim = paths.concat( opt.dirModel, 'optimState_%03d.t7' )
	opt.pathTrainLog = paths.concat( opt.dirModel, 'train.log' )
	opt.pathValLog = paths.concat( opt.dirModel, 'val.log' )
	opt.pathTestLog = paths.concat( opt.dirModel, 'test_%d.log' )
	-- Value processing.
	opt.learnRate = opt.learnRate:split( ',' )
	for k,v in pairs( opt.learnRate ) do opt.learnRate[ k ] = tonumber( v ) end
	return opt
end
function task:createDbTrain(  )
	local dbtr = {  }
	dbtr.vid2path,
	dbtr.vid2numim,
	dbtr.vid2cid,
	dbtr.cid2name,
	dbtr.frameFormat = genDb( 'train' )
	local numVideo = dbtr.vid2path:size( 1 )
	local numClass = dbtr.cid2name:size( 1 )
	self:print( string.format( 'Train: %d videos, %d classes.', numVideo, numClass ) )
	-- Verification.
	assert( dbtr.vid2path:size( 1 ) == dbtr.vid2numim:numel(  ) )
	assert( dbtr.vid2path:size( 1 ) == dbtr.vid2cid:numel(  ) )
	assert( dbtr.cid2name:size( 1 ) == dbtr.vid2cid:max(  ) )
	return dbtr
end
function task:createDbVal(  )
	local dbval = {  }
	dbval.vid2path,
	dbval.vid2numim,
	dbval.vid2cid,
	dbval.cid2name,
	dbval.frameFormat = genDb( 'val' )
	local numVideo = dbval.vid2path:size( 1 )
	local numClass = dbval.cid2name:size( 1 )
	self:print( string.format( 'Val: %d videos, %d classes.', numVideo, numClass ) )
	-- Verification.
	assert( dbval.vid2path:size( 1 ) == dbval.vid2numim:numel(  ) )
	assert( dbval.vid2path:size( 1 ) == dbval.vid2cid:numel(  ) )
	assert( dbval.cid2name:size( 1 ) == dbval.vid2cid:max(  ) )
	return dbval
end
function task:setNumBatch(  )
	local batchSize = self.opt.batchSize
	local numFlow = self.opt.numFlow
	local numBatchTrain = math.floor( self.dbtr.vid2path:size( 1 ) * numFlow / batchSize )
	local numBatchVal = math.floor( self.dbval.vid2path:size( 1 ) * numFlow / batchSize )
	return numBatchTrain, numBatchVal
end
function task:setNumQuery(  )
	return self.dbval.vid2path:size( 1 )
end
function task:estimateInputStat(  )
	if self.opt.caffeInput then
		return { mean = torch.Tensor{ 128, 128 }, std = torch.Tensor{ 0, 0 } }
	else
		return { mean = torch.Tensor{ 0.5, 0.5 }, std = torch.Tensor{ 0, 0 } }
	end
end
function task:setModelSpecificOption(  )
	self.opt.cropSize = 224
	self.opt.keepAspect = true
	self.opt.normalizeStd = false
	self.opt.caffeInput = true
	self.opt.numOut = 1
end
function task:defineModel(  )
	require 'loadcaffe'
	-- Get params.
	local numGpu = self.opt.numGpu
	local cropSize = self.opt.cropSize
	local numFlow = self.opt.numFlow
	local numClass = self.dbtr.cid2name:size( 1 )
	local dropout = self.opt.dropout
	local proto = gpath.net.vgg16_caffe_proto
	local caffemodel = gpath.net.vgg16_caffe_model
	-- Check options.
	assert( cropSize == 224 )
	assert( self.opt.keepAspect )
	assert( not self.opt.normalizeStd )
	assert( self.opt.caffeInput )
	assert( self.opt.numOut == 1 )
	assert( numFlow >= 1 )
	assert( dropout >= 0 and dropout <= 1 )
	-- Create model.
	self:print( 'Load pre-trained Caffe feature.' )
	local features = loadcaffe.load( proto, caffemodel, self.opt.backend )
	features:remove( 40 ) -- removes softmax.
	features:remove( 39 ) -- removes fc.
	features:remove( 38 ) -- removes dropout.
	features:remove( 35 ) -- removes dropout.
	features:insert( nn.Dropout( dropout ), 35 )
	features:add( nn.Dropout( dropout ) )
	local conv1 = cudnn.SpatialConvolution( 2 * numFlow, 64, 3, 3, 1, 1, 1, 1, 1 ):cuda(  )
	local srcw, srcg = features.modules[ 1 ]:parameters(  )
	local dstw, dstg = conv1:parameters(  )
	dstw[ 1 ]:copy( srcw[ 1 ]:mean( 2 ):repeatTensor( 1, 2 * numFlow, 1, 1 ):mul( 3 / ( 2 * numFlow ) ) )
	dstw[ 2 ]:copy( srcw[ 2 ] )
	dstg[ 1 ]:copy( srcg[ 1 ]:mean( 2 ):repeatTensor( 1, 2 * numFlow, 1, 1 ):mul( 3 / ( 2 * numFlow ) ) )
	dstg[ 2 ]:copy( srcg[ 2 ] )
	features:remove( 1 )
	features:insert( conv1, 1 )
	features:insert( nn.View( -1, 2 * numFlow, cropSize, cropSize ), 1 )
	features:cuda(  )
	local classifier = nn.Sequential(  )
	classifier:add( nn.Linear( 4096, numClass ) )
	classifier:add( nn.LogSoftMax(  ) )
	classifier:cuda(  )
	local model = nn.Sequential(  )
	model:add( features )
	model:add( classifier )
	model:cuda(  )
	-- Wrap up net with data parallel table if needed.
	model = makeDataParallel( model, numGpu )
	return model
end
function task:defineCriterion(  )
	return nn.ClassNLLCriterion(  )
end
function task:groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	if self.opt.numGpu > 1 then
		params[ 1 ], grads[ 1 ] = model.modules[ 1 ].modules[ 1 ]:getParameters(  ) -- Features.
		params[ 2 ], grads[ 2 ] = model.modules[ 1 ].modules[ 2 ]:getParameters(  ) -- Classifier.
	else
		params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Features.
		params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  ) -- Classifier.
	end
	optims[ 1 ] = { -- Features.
		learningRate = self.opt.learnRate[ 1 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 2 ] = { -- Classifier.
		learningRate = self.opt.learnRate[ 2 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	return params, grads, optims
end
function task:changeModelTrain( model )
	-- Nothing to change.
end
function task:changeModelVal( model )
	-- Nothing to change.
end
function task:changeModelTest( model )
	-- Nothing to change.
end
function task:getBatchTrain(  )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local numFlow = self.opt.numFlow
	assert( ( batchSize / self.opt.numGpu ) % numFlow == 0 )
	local input = torch.Tensor( batchSize, 2, cropSize, cropSize )
	local label = torch.LongTensor( batchSize / numFlow )
	local numVideo = self.dbtr.vid2path:size( 1 )
	local cnt = 0
	for v = 1, batchSize / numFlow do
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.dbtr.vid2path[ vid ] ) )
		local numFrame = self.dbtr.vid2numim[ vid ]
		local cid = self.dbtr.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - numFlow + 1 ) )
		local rw = torch.uniform(  )
		local rh = torch.uniform(  )
		local rf = torch.uniform(  )
		for f = 1, numFlow do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbtr.frameFormat, fid ) )
			cnt = cnt + 1
			input[ cnt ]:copy( self:processImageTrain( fpath, rw, rh, rf ) )
		end
		label[ v ] = cid
	end
	return input, label
end
function task:getBatchVal( fidStart )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local numFlow = self.opt.numFlow
	local vidStart = ( fidStart - 1 ) / numFlow + 1
	assert( ( batchSize / self.opt.numGpu ) % numFlow == 0 )
	local input = torch.Tensor( batchSize, 2, cropSize, cropSize )
	local label = torch.LongTensor( batchSize / numFlow )
	local cnt = 0
	for v = 1, batchSize / numFlow do
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
		local numFrame = self.dbval.vid2numim[ vid ]
		local cid = self.dbval.vid2cid[ vid ]
		local startFrame = math.floor( math.max( 0, numFrame - numFlow ) / 2 ) + 1
		for f = 1, numFlow do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
			cnt = cnt + 1
			input[ cnt ]:copy( self:processImageVal( fpath ) )
		end
		label[ v ] = cid
	end
	return input, label
end
function task:evalBatch( output, label )
	local numVideo = self.opt.batchSize / self.opt.numFlow
	assert( numVideo == output:size( 1 ) )
	assert( numVideo == label:numel(  ) )
	local _, rank2cid = output:float(  ):sort( 2, true )
	local top1 = 0
	for v = 1, numVideo do
		if rank2cid[ v ][ 1 ] == label[ v ] then
			top1 = top1 + 1
		end
	end
	return torch.Tensor{ top1 * 100 / numVideo }
end
function task:getQuery( queryNumber )
	local augments = torch.Tensor{
			  { 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.5 },
			  { 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.5 },
			  { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 } }
	augments = augments[ { {  }, { 5 } } ]:cat( augments[ { {  }, { 10 } } ], 2 )
	local numAugment = augments:size( 2 )
	local stride = 1
	local cropSize = self.opt.cropSize
	local vid = queryNumber
	local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
	local numFrame = self.dbval.vid2numim[ vid ]
	local numSeq = math.floor( numFrame / stride ) + 1
	local query = torch.Tensor( numSeq * numAugment, 2, cropSize, cropSize )
	local fcnt = 0
	for seq = 1, numSeq do
		local startFrame = 1 + stride * ( seq - 1 )
		for a = 1, numAugment do
			local rw = augments[ 1 ][ a ]
			local rh = augments[ 2 ][ a ]
			local rf = augments[ 3 ][ a ]
			local fid = math.min( numFrame, startFrame )
			local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
			fcnt = fcnt + 1
			query[ fcnt ]:copy( self:processImageTrain( fpath, rw, rh, rf ) )
		end
	end
	assert( query:size( 1 ) % self.opt.batchSize % self.opt.numGpu == 0 )
	return query
end
function task:aggregateAnswers( answers )
	for k, v in pairs( answers ) do
		answers[ k ] = v:mean( 1 )
	end
	return answers
end
function task:evaluate( answers, qids )
	local numQuery = answers[ 1 ]:size( 1 )
	local numClass = self.dbval.cid2name:size( 1 )
	local scores = {  }
	assert( qids:numel(  ) == numQuery )
	assert( qids:max(  ) == numQuery )
	assert( self.dbval.vid2path:size( 1 ) == numQuery )
	for k, v in pairs( answers ) do
		local pathTestLog = self.opt.pathTestLog:format( k )
		local testLogger = io.open( pathTestLog, 'w' )
		testLogger:write( 'QUERY-LEVEL EVALUATION\n' )
		print( 'QUERY-LEVEL EVALUATION' )
		local qid2top1 = torch.Tensor( numQuery ):fill( 0 )
		local cid2num = torch.Tensor( numClass ):fill( 0 )
		local cid2top1 = torch.Tensor( numClass ):fill( 0 )
		local _, pcids = v:float(  ):sort( 2, true )
		for q = 1, numQuery do
			local qid = qids[ q ]
			local pcid = pcids[ q ][ 1 ]
			local cid = self.dbval.vid2cid[ qid ]
			local vpath = ffi.string( torch.data( self.dbval.vid2path[ qid ] ) )
			local score = 0
			if pcid == cid then score = 1 end
			qid2top1[ qid ] = score
			cid2top1[ cid ] = cid2top1[ cid ] + score
			cid2num[ cid ] = cid2num[ cid ] + 1
			testLogger:write( string.format( 'QID %06d SCORE %.2f PRED %06d GT %06d PATH %s\n',
			qid, score, pcid, cid, vpath ) )
		end
		testLogger:write( string.format( 'MEAN TOP1 %.2f\n', qid2top1:mean(  ) * 100 ) )
		print( string.format( 'MEAN TOP1 %.2f', qid2top1:mean(  ) * 100 ) )
		testLogger:write( 'CLASS-LEVEL EVALUATION\n' )
		print( 'CLASS-LEVEL EVALUATION' )
		cid2top1:cdiv( cid2num )
		for cid = 1, numClass do
			local cname = ffi.string( torch.data( self.dbval.cid2name[ cid ] ) )
			local score = cid2top1[ cid ] * 100
			testLogger:write( string.format( 'CID %03d SCORE %.2f CNAME %s\n', cid, score, cname ) )
			print( string.format( 'CID %03d SCORE %.2f CNAME %s', cid, score, cname ) )
		end
		testLogger:write( string.format( 'MEAN CLASS SCORE %.2f', cid2top1:mean(  ) * 100 ) )
		print( string.format( 'MEAN CLASS SCORE %.2f', cid2top1:mean(  ) * 100 ) )
		testLogger:close(  )
	end
end
--------------------------------------------------
-------- TASK-SPECIFIC INTERNAL FUNCTIONS --------
--------------------------------------------------
require 'image'
function task:processImageTrain( path, rw, rh, rf )
	collectgarbage(  )
	local input = self:loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do random crop.
	local oW = self.opt.cropSize
	local oH = self.opt.cropSize
	local h1 = math.ceil( ( iH - oH ) * rh )
	local w1 = math.ceil( ( iW - oW ) * rw )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Do horz-flip.
	if rf > 0.5 then
		out = image.hflip( out )
		local maxval = 1
		if self.opt.caffeInput then maxval = 255 end
		out[ 1 ]:mul( -1 ):add( maxval )
	end
	-- Normalize.
	out = self:normalizeImage( out )
	return out
end
function task:processImageVal( path )
	collectgarbage(  )
	local input = self:loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do central crop.
	local oW = self.opt.cropSize
	local oH = self.opt.cropSize
	local h1 = math.ceil( ( iH - oH ) / 2 )
	local w1 = math.ceil( ( iW - oW ) / 2 )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Normalize.
	out = self:normalizeImage( out )
	return out
end
function task:resizeImage( im )
	local imageSize = self.opt.imageSize
	if self.opt.keepAspect then
		if im:size( 3 ) < im:size( 2 ) then
			im = image.scale( im, imageSize, imageSize * im:size( 2 ) / im:size( 3 ) )
		else
			im = image.scale( im, imageSize * im:size( 3 ) / im:size( 2 ), imageSize )
		end
	else
		im = image.scale( im, imageSize, imageSize )
	end
	return im
end
function task:loadImage( path )
	local imu = image.load( path, 1, 'float' )
	local pathv = path:gsub( '/u/', '/v/' )
	local imv = image.load( pathv, 1, 'float' )
	local im = torch.cat( imu, imv, 1 )
	im = self:resizeImage( im )
	if self.opt.caffeInput then
		im:mul( 255 )
	end
	return im
end
function task:normalizeImage( im )
	for i = 1, 2 do
		if self.inputStat.mean then im[ i ]:add( -self.inputStat.mean[ i ] ) end
		if self.inputStat.std and self.opt.normalizeStd then im[ i ]:div( self.inputStat.std[ i ] ) end
	end
	return im
end
