-- Customized from fb.resnet.torch/datasets/transforms.lua
require 'image'
local M = {}
local function blend( img1, img2, alpha )
	return img1:mul( alpha ):add( 1 - alpha, img2 )
end
local function grayscale( dst, img )
	dst:resizeAs( img )
	dst[ 1 ]:zero(  )
	dst[ 1 ]:add( 0.299, img[ 1 ] ):add( 0.587, img[ 2 ] ):add( 0.114, img[ 3 ] )
	dst[ 2 ]:copy( dst[ 1 ] )
	dst[ 3 ]:copy( dst[ 1 ] )
	return dst
end
local function Saturation( img, alpha )
	local gs = img.new(  )
	grayscale( gs, img )
	blend( img, gs, alpha )
	return img
end
local function Brightness( img, alpha )
	local gs = img.new(  )
	gs:resizeAs( img ):zero(  )
	blend( img, gs, alpha )
	return img
end
local function Contrast( img, alpha )
	local gs = img.new(  )
	grayscale( gs, img )
	gs:fill( gs[ 1 ]:mean(  ) )
	blend( img, gs, alpha )
	return img
end
function M.ColorNormalize( input, mean, std )
	input = input:clone(  )
	for i=1,3 do
		input[ i ]:add( -mean[ i ] )
		input[ i ]:div( std[ i ] )
	end
	return input
end
function M.Scale( input, size )
	local w, h = input:size( 3 ), input:size( 2 )
	if ( w <= h and w == size ) or ( h <= w and h == size ) then
		return input
	end
	if w < h then
		return image.scale( input, size, h/w * size, 'bicubic' )
	else
		return image.scale( input, w/h * size, size, 'bicubic' )
	end
end
function M.RandomCrop( input, size, rw, rh )
	local w, h = input:size( 3 ), input:size( 2 )
	if w == size and h == size then
		return input
	end
	local x1, y1 = torch.round( ( w - size ) * rw ), torch.round( ( h - size ) * rh )
	local out = image.crop( input, x1, y1, x1 + size, y1 + size )
	assert( out:size( 2 ) == size and out:size( 3 ) == size, 'wrong crop size' )
	return out
end
function M.RandomSizedCrop( input, size, randn )
	local attempt = 0
	repeat
		local area = input:size( 2 ) * input:size( 3 )
		local targetArea = ( 0.08 + ( 1.0 - 0.08 ) * randn[ 1 ] ) * area
		local aspectRatio = 3/4 + ( 4/3 - 3/4 ) * randn[ 2 ]
		local w = torch.round( math.sqrt( targetArea * aspectRatio ) )
		local h = torch.round( math.sqrt( targetArea / aspectRatio ) )
		if randn[ 3 ] < 0.5 then
			w, h = h, w
		end
		if h <= input:size( 2 ) and w <= input:size( 3 ) then
			local y1 = torch.round( ( input:size( 2 ) - h ) * randn[ 4 ] )
			local x1 = torch.round( ( input:size( 3 ) - w ) * randn[ 5 ] )
			local out = image.crop( input, x1, y1, x1 + w, y1 + h )
			assert( out:size( 2 ) == h and out:size( 3 ) == w, 'wrong crop size' )
			return image.scale( out, size, size, 'bicubic' )
		end
		local randn_ = {  }
		for i = 1,5 do randn_[ ( i )%5+1 ] = randn[ i ] end
		randn = randn_
		attempt = attempt + 1
	until attempt >= 5
	return M.RandomCrop( M.Scale( input, size ), size, 0.5, 0.5 )
end
function M.HorizontalFlip( input, prob )
	if prob < 0.5 then
		input = image.hflip( input )
	end
	return input
end
function M.Lighting( input, alpha, eigval, eigvec )
	local rgb = eigvec:clone(  )
		:cmul( alpha:view( 1, 3 ):expand( 3, 3 ) )
		:cmul( eigval:view( 1, 3 ):expand( 3, 3 ) )
		:sum( 2 )
		:squeeze(  )
	input = input:clone(  )
	for i=1,3 do
		input[ i ]:add( rgb[ i ] )
	end
	return input
end
function M.ColorJitter( input, order, alpha )
	for i=1,3 do
		if order[ i ] == 1 then
			input = Brightness( input, alpha[ 1 ] )
		elseif order[ i ] == 2 then
			input = Contrast( input, alpha[ 2 ] )
		elseif order[ i ] == 3 then
			input = Saturation( input, alpha[ 3 ] )
		end
	end
	return input
end
return M
