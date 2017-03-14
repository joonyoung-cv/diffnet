torch.setdefaulttensortype( 'torch.FloatTensor' )
paths.dofile( 'util.lua' )
paths.dofile( 'setpath.lua' )

-- Define task.
assert( arg[ 1 ] == '-task', 'Specify a defined task name.' )
local taskFile = paths.concat( 'task', arg[ 2 ] .. '.lua' )
paths.dofile( taskFile )

-- Set task manager.
local task = TaskManager(  )
task:setOption( arg )
task:setDb(  )
task:setInputStat(  )

-- Get necessary data. 
local opt = task:getOption(  )
local funtr1, funtr2, funtr3 = task:getFunctionTrain(  )
local funval1, funval2, funval3 = task:getFunctionVal(  )
local numbtr, numbval = task:getNumBatch(  )

-- Hire donkeys working for data loading.
-- This is modified from Soumith's data.lua.
local Threads = require 'threads'
local donkeys = {  }
Threads.serialization( 'threads.sharedserialize' )
if opt.numDonkey > 0 then
	donkeys = Threads(
		opt.numDonkey,
		function(  )
			paths.dofile( taskFile )
		end,
		function( tid )
			local seed = tid
			torch.manualSeed( seed )
			torch.setnumthreads( 1 )
			print( string.format( 'DONKEY) Start donkey %d with seed %d.', tid, seed ) )
		end
	)
else
	function donkeys:addjob( f1, f2 ) f2( f1(  ) ) end
	function donkeys:synchronize(  ) end
	torch.manualSeed( se )
end
donkeys:synchronize(  ) 

-- Set val.
val = paths.dofile( 'val.lua' )
val.setOption( opt, numbval )
val.pathValLog = val.pathValLog:match( '(.+).log' ) .. '_only.log'
val.setDonkey( donkeys )
val.setFunction( funval1, funval2, funval3 )

-- Do the job.
for e = 1, opt.numEpoch do
	local pathModel = task.opt.pathModel
	local numGpu = task.opt.numGpu
	local backend = task.opt.backend
	print( string.format( 'Load model from epoch %d.', e ) )
	local model = loadDataParallel( pathModel:format( e ), numGpu, backend )
	local criterion = task:defineCriterion(  )
	model:cuda(  )
	criterion:cuda(  )
	cutorch.setDevice( 1 )
	print( 'Done.' )
	val.setModel( { model = model, criterion = criterion } )
	val.val( e )
end
