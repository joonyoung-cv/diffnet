require 'paths'
require 'sys'
local ffi = require 'ffi'
local function strTableToTensor( strTable )
	local maxStrLen = 0
	local numStr = #strTable
	for _, path in pairs( strTable ) do
		if maxStrLen < path:len(  ) then maxStrLen = path:len(  ) end
	end
	maxStrLen = maxStrLen + 1
	local charTensor = torch.CharTensor( numStr, maxStrLen ):fill( 0 )
	local pt = charTensor:data(  )
	for _, path in pairs( strTable ) do
		ffi.copy( pt, path )
		pt = pt + maxStrLen
	end
	collectgarbage(  )
	return charTensor
end
local labelPostfix = '_test_split3.txt'
function genDb( setName )
	local rootDir = gpath.db.hmdb51_rgb_s3
	local split
	if setName == 'train' then
		split = 1
	elseif setName == 'val' then
		split = 2
	end
	local labelDir = paths.concat( rootDir, 'testTrainMulti_7030_splits' )
	local videoDir = paths.concat( rootDir, 'jpegs_256' )
	-- Make cid2name.
	local cid2name = {  }
	local p = io.popen( 'ls ' .. labelDir )
	for line in p:lines(  ) do
		local cname = line:match( '(.+)' .. labelPostfix )
		if cname then cid2name[ #cid2name + 1 ] = cname end
	end
	p:close(  )
	-- Make vid2path, vid2cid, vid2numim.
	local vid, vid2path, vid2cid, vid2numf = 0, {  }, {  }, {  }
	for cid, cname in pairs( cid2name ) do
		local labelFname = paths.concat( labelDir, cname .. labelPostfix )
		for line in io.lines( labelFname ) do
			local vname, spl = line:match( '(.+)%.avi%s(%d)' )
			assert( vname and spl )
			if tonumber( spl ) == split then
				local vpath = paths.concat( videoDir, vname )
				local numf = 0
				for f in paths.files( vpath ) do if f:match( '.jpg' ) then numf = numf + 1 end end
				vid = vid + 1
				vid2path[ vid ] = vpath
				vid2numf[ vid ] = numf
				vid2cid[ vid ] = cid
				if vid % 100 == 0 then print( vid .. ' videos found for ' .. setName .. '.' ) end
				assert( vpath:len(  ) > 0 and numf > 2 )
			end
		end
	end
	print( #vid2path .. ' videos found for ' .. setName .. ' in total.' )
	-- Convert tables to tensors.
	local vid2path = strTableToTensor( vid2path )
	local cid2name = strTableToTensor( cid2name )
	vid2numf = torch.LongTensor( vid2numf )
	vid2cid = torch.LongTensor( vid2cid )
	collectgarbage(  )
	return vid2path, vid2numf, vid2cid, cid2name, 'frame%06d.jpg'
end
