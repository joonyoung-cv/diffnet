require 'image'
local ffi = require 'ffi'
local ip = require 'improc'
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
	self.inputStat = self:estimateInputStat(  )
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
	local cudnnMode = self.opt.cudnn
	local startEpoch = 1
	require 'cudnn'
	if cudnnMode == 'fastest' then
		cudnn.fastest = true
		cudnn.benchmark = true
 	end
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
		cudnn.convert( model, cudnn )
		params, grads, optims = self:groupParams( model )
	elseif startEpoch == 1 and startFrom:len(  ) > 0 then
		self:print( 'Load user-defined model.' .. startFrom )
		model = loadDataParallel( startFrom, numGpu )
		params, grads, optims = self:groupParams( model )
	elseif startEpoch > 1 then
		self:print( string.format( 'Load model from epoch %d.', startEpoch - 1 ) )
		model = loadDataParallel( pathModel:format( startEpoch - 1 ), numGpu )
		params, grads, _ = self:groupParams( model )
		optims = torch.load( pathOptim:format( startEpoch - 1 ) )
	end
	self:print( 'Done.' )
	local criterion = self:defineCriterion(  )
	self:print( 'Model looks' )
	print( model )
	print( criterion )
	self:print( 'Put net on gpu.' )
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
	cmd:option( '-numDonkey', 16, 'Number of donkeys for data loading.' )
	cmd:option( '-cudnn', 'fastest', 'fastest | default' )
	-- Data.
	cmd:option( '-data', 'UCF101', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 256, 'Short side of initial resize.' )
	-- Train.
	cmd:option( '-numEpoch', 50, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 75, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 256, 'Frame-level mini-batch size.' )
	cmd:option( '-learnRate', '1e-2,1e-2', 'Supports multi-lr for multi-module like "lr1,lr2,lr3".' )
	cmd:option( '-momentum', 0.9, 'Momentum.' )
	cmd:option( '-weightDecay', 1e-4, 'Weight decay.' )
	cmd:option( '-startFrom', '', 'Path to the initial model. Using it for LR decay is recommended.' )
	local opt = cmd:parse( arg or {  } )
	-- Set dst paths.
	local dirRoot = paths.concat( gpath.dataout, opt.data )
	local pathDbTrain = paths.concat( dirRoot, 'dbTrain.t7' )
	local pathDbVal = paths.concat( dirRoot, 'dbVal.t7' )
	local ignore = { numGpu=true, numDonkey=true, data=true, numEpoch=true, startFrom=true }
	local dirModel = paths.concat( dirRoot, cmd:string( opt.task, opt, ignore ) )
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match( '(.+)/model_(%d+).t7' )
		dirModel = paths.concat( baseDir, cmd:string( 'model_' .. epoch, opt, ignore ) )
	end
	opt.dirRoot = dirRoot
	opt.pathDbTrain = pathDbTrain
	opt.pathDbVal = pathDbVal
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
	local numBatchTrain = math.floor( self.dbtr.vid2path:size( 1 ) / batchSize )
	local numBatchVal = math.floor( self.dbval.vid2path:size( 1 ) / batchSize )
	return numBatchTrain, numBatchVal
end
function task:setNumQuery(  )
	return self.dbval.vid2path:size( 1 )
end
function task:estimateInputStat(  )
	local mean = { 0.485, 0.456, 0.406 }
	local std = { 0.229, 0.224, 0.225 }
	local eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 }
	local eigvec = torch.Tensor{
			  { -0.5675,  0.7192,  0.4009 },
			  { -0.5808, -0.0045, -0.8140 },
			  { -0.5836, -0.6948,  0.4203 } }
	return { mean = mean, std = std, eigval = eigval, eigvec = eigvec }
end
function task:setModelSpecificOption(  )
	self.opt.cropSize = 224
	self.opt.numOut = 1
end
function task:defineModel(  )
	require 'cudnn'
	require 'cunn'
	-- Get params.
	local numGpu = self.opt.numGpu
	local batchSize = self.opt.batchSize
	local numClass = self.dbtr.cid2name:size( 1 )
	local inputSize = self.opt.cropSize
	-- Check options.
	assert( self.opt.cropSize == 224 )
	assert( self.opt.numOut == 1 )
	assert( batchSize % numGpu == 0 )
	assert( ( batchSize / numGpu ) % 1 == 0 )
	-- Make initial model.
	local features_ = torch.load( gpath.net.res18_torch_model )
	features_:remove(  ) -- removes classifier.
	local features = nn.Sequential(  )
	for l = 1, #features_.modules do
		local module = features_.modules[ l ]
		if torch.type( module ) == 'nn.Sequential' then
			for m = 1, #module.modules do
				features:add( module.modules[ m ] )
			end
		else
			features:add( module )
		end
	end
	local features = features_
	features:cuda(  )
	local classifier = nn.Sequential(  )
	local linear = nn.Linear( 512, numClass )
	linear.bias:zero(  )
	classifier:add( linear )
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
		nesterov = true,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 2 ] = { -- Classifier.
		learningRate = self.opt.learnRate[ 2 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		nesterov = true,
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
	local input = torch.Tensor( batchSize, 3, cropSize, cropSize )
	local numVideo = self.dbtr.vid2path:size( 1 )
	local eigval = self.inputStat.eigval
	local eigvec = self.inputStat.eigvec
	local mean = self.inputStat.mean
	local std = self.inputStat.std
	local label = torch.LongTensor( batchSize )
	for v = 1, batchSize do
		collectgarbage(  )
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.dbtr.vid2path[ vid ] ) )
		local numFrame = self.dbtr.vid2numim[ vid ]
		local cid = self.dbtr.vid2cid[ vid ]
		local fid = torch.random( 1, numFrame )
		local randn, alpha = {  }, {  }
		for i = 1, 5 do randn[ i ] = torch.uniform(  ) end
		for i = 1, 3 do alpha[ i ] = 1.0 + torch.uniform( -0.4, 0.4 ) end
		local alphaLight = torch.Tensor( 3 ):normal( 0, 0.1 ):float(  )
		local order = torch.randperm( 3 ):totable(  )
		local prob = torch.uniform(  )
		local fpath = paths.concat( vpath, string.format( self.dbtr.frameFormat, fid ) )
		local im = image.load( fpath, 3, 'float' )
		im = ip.RandomSizedCrop( im, cropSize, randn )
		im = ip.ColorJitter( im, order, alpha )
		im = ip.Lighting( im, alphaLight, eigval, eigvec )
		im = ip.ColorNormalize( im, mean, std )
		im = ip.HorizontalFlip( im, prob )
		input[ v ]:copy( im )
		label[ v ] = cid
	end
	return input, label
end
function task:getBatchVal( vidStart )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local imageSize = self.opt.imageSize
	local input = torch.Tensor( batchSize, 3, cropSize, cropSize )
	local mean = self.inputStat.mean
	local std = self.inputStat.std
	local label = torch.LongTensor( batchSize )
	for v = 1, batchSize do
		collectgarbage(  )
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
		local numFrame = self.dbval.vid2numim[ vid ]
		local cid = self.dbval.vid2cid[ vid ]
		local fid = math.max( 1, math.floor( numFrame / 2 ) )
		local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
		local im = image.load( fpath, 3, 'float' )
		im = ip.Scale( im, imageSize )
		im = ip.ColorNormalize( im, mean, std )
		im = ip.RandomCrop( im, cropSize, 0.5, 0.5 )
		input[ v ]:copy( im )
		label[ v ] = cid
	end
	return input, label
end
function task:evalBatch( output, label )
	local batchSize = self.opt.batchSize
	local numVideo = output:size( 1 )
	assert( numVideo == batchSize )
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
	local imageSize = self.opt.imageSize
	local mean = self.inputStat.mean
	local std = self.inputStat.std
	local vid = queryNumber
	local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
	local numFrame = self.dbval.vid2numim[ vid ]
	local numSeq = math.floor( numFrame / stride ) + 1
	local query = torch.Tensor( numSeq * numAugment, 3, cropSize, cropSize )
	local fcnt = 0
	for seq = 1, numSeq do
		local fid = 1 + stride * ( seq - 1 )
		local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
		local im_ = image.load( fpath, 3, 'float' )
		for a = 1, numAugment do
			collectgarbage(  )
			local rw = augments[ 1 ][ a ]
			local rh = augments[ 2 ][ a ]
			local rf = augments[ 3 ][ a ]
			local im = im_:clone(  )
			im = ip.Scale( im, imageSize )
			im = ip.ColorNormalize( im, mean, std )
			im = ip.RandomCrop( im, cropSize, rw, rh )
			im = ip.HorizontalFlip( im, rf )
			fcnt = fcnt + 1
			query[ fcnt ]:copy( im )
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
