local val = {  }
val.inputs = torch.CudaTensor(  )
val.labels = torch.CudaTensor(  )
val.netTimer = torch.Timer(  )
val.dataTimer = torch.Timer(  )
function val.setOption( opt, numBatchVal )
	assert( numBatchVal > 0 )
	assert( numBatchVal % 1 == 0 )
	assert( opt.batchSize > 0 )
	assert( opt.numOut > 0 )
	assert( opt.pathValLog:match( '(.+).log$' ):len(  ) > 0 )
	val.batchSize = opt.batchSize
	val.numOut = opt.numOut
	val.pathValLog = opt.pathValLog
	val.epochSize = numBatchVal
end
function val.setModel( modelSet )
	val.model = modelSet.model
	val.criterion = modelSet.criterion
end
function val.setDonkey( donkeys )
	val.donkeys = donkeys
end
function val.setFunction( changeModel, getBatch, evalBatch )
	val.changeModel = changeModel
	val.getBatch = getBatch
	val.evalBatch = evalBatch
end
function val.val( epoch )
	-- Initialization.
	local epochTimer = torch.Timer(  )
	local getBatch = val.getBatch
	local valBatch = val.valBatch
	val.epoch = epoch
	val.evalEpoch = torch.Tensor( val.numOut ):fill( 0 )
	val.lossEpoch = 0
	val.numEvalEpoch = 0
	val.batchNumber = 0
	-- Do the job.
	val.print( string.format( 'Validation epoch %d.', epoch ) )
	cutorch.synchronize(  )
	val.changeModel( val.model )
	val.model:evaluate(  )
	for b = 1, val.epochSize do
		local s = ( b - 1 ) * val.batchSize + 1
		val.donkeys:addjob(
			function(  )
				return getBatch( s )
			end, -- Job callback.
			function( x, y )
				valBatch( x, y )
			end -- End callback.
		)
	end
	val.donkeys:synchronize(  )
	cutorch.synchronize(  )
	val.evalEpoch = val.evalEpoch / val.numEvalEpoch
	val.lossEpoch = val.lossEpoch / val.numEvalEpoch
	local evalEpochStr = val.tensor2str( val.evalEpoch, '%.4f' )
	val.print( string.format( 'Epoch %d, time %.2fs, avg loss %.4f, eval %s', 
		epoch, epochTimer:time(  ).real, val.lossEpoch, evalEpochStr ) )
	if epoch > 0 then
		local valLogger = io.open( val.pathValLog, 'a' )
		valLogger:write( string.format( '%03d %.4f %s\n', epoch, val.lossEpoch, evalEpochStr ) )
		valLogger:close(  )
	end
	collectgarbage(  )
end
function val.valBatch( inputsCpu, labelsCpu )
	-- Initialization.
	local numEval = labelsCpu:size( 1 )
	local dataTime = val.dataTimer:time(  ).real
	val.netTimer:reset(  )
	val.inputs:resize( inputsCpu:size(  ) ):copy( inputsCpu )
	val.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	-- Forward pass.
	cutorch.synchronize(  )
	local outputs = val.model:forward( val.inputs )
	local err = val.criterion:forward( outputs, val.labels )
	cutorch.synchronize(  )
	val.batchNumber = val.batchNumber + 1
	val.lossEpoch = val.lossEpoch + err * numEval
	-- Task evaluation.
	local eval = val.evalBatch( outputs, labelsCpu )
	val.evalEpoch = val.evalEpoch + eval * numEval
	val.numEvalEpoch = val.numEvalEpoch + numEval
	local evalStr = val.tensor2str( eval, '%.2f' )
	local netTime = val.netTimer:time(  ).real
	local totalTime = dataTime + netTime
	local speed = val.batchSize / totalTime
	-- Print information.
	val.print( string.format( 'Epoch %d, %d/%d, %dim/s (data %.2fs, net %.2fs), err %.2f, eval %s', 
		val.epoch, val.batchNumber, val.epochSize, speed, dataTime, netTime, err, evalStr ) )
	val.dataTimer:reset(  )
	collectgarbage(  )
end
function val.tensor2str( tensor, precision )
	local str = ''
	for i = 1, tensor:numel(  ) do
		str = string.format( '%s' .. precision .. ' ', str, tensor[ i ] )
	end
	return str
end
function val.print( str )
	print( 'VAL) ' .. str )
end
return val
