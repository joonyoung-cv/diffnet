-- Customized from fb.resnet.torch
require 'nn'
require 'cunn'
require 'cudnn'
local function deepCopy( tbl )
	local copy = {  }
	for k, v in pairs( tbl ) do
		if type( v ) == 'table' then
			copy[ k ] = deepCopy( v ) else
			copy[ k ] = v 
 		end
	end
	if torch.typename( tbl ) then
		torch.setmetatable( copy, torch.typename( tbl ) )
	end
	return copy
end
function makeDataParallel( model, numGpu )
	if numGpu > 1 then
		print( string.format( 'Wrap up model with data parallel table. (X%s)', numGpu ) )
		local gpus = torch.range( 1, numGpu ):totable(  )
		local fastest, benchmark = cudnn.fastest, cudnn.benchmark
		local dpt = nn.DataParallelTable( 1, true, true )
			:add( model, gpus )
			:threads( function(  )
				local cudnn = require 'cudnn'
				cudnn.fastest, cudnn.benchmark = fastest, benchmark end )
		dpt.gradInput = nil
		model = dpt
	end
   return model
end
function saveDataParallel( filename, model )
	if torch.type( model ) == 'nn.DataParallelTable' then model = model:get( 1 ) end
	model = deepCopy( model ):float(  ):clearState(  )
	torch.save( filename, model )
end
function loadDataParallel( filename, numGpu )
   local model = torch.load( filename ):cuda(  )
	if torch.type( model ) == 'nn.DataParallelTable' then model = model:get( 1 ) end
	model.__memoryOptimized = nil
	return makeDataParallel( model, numGpu )
end
