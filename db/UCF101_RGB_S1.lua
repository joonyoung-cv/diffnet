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
local function readLinesFrom( fpath )
	lines = {  }
	for line in io.lines( fpath ) do
		lines[ #lines + 1 ] = line
	end
	return lines
end
function genDb( setName )
	local rootDir = gpath.db.ucf101_rgb_s1
	local labelDir = paths.concat( rootDir, 'ucfTrainTestlist' )
	local videoDir = paths.concat( rootDir, 'jpegs_256' )
	-- Make cid2name.
	local cid2name = readLinesFrom( paths.concat( labelDir, 'classInd.txt' ) )
	local cname2id = {  }
	for cid, cname in pairs( cid2name ) do
		cid2name[ cid ] = cid2name[ cid ]:match( '%d+%s+(.+)%s' )
		cname2id[ cid2name[ cid ] ] = cid
	end
	-- Make vid2path, vid2cid, vid2numim.
	local labelFname, parse
	if setName == 'train' then
		labelFname = 'trainlist01.txt'
		parse = function( str )
			local vpath, cid = str:match( '.+/(.+)%.avi%s(%d+)' )
			return vpath, cid
		end
	elseif setName == 'val' then
		labelFname = 'testlist01.txt'
		parse = function( str )
			local cname, vpath = str:match( '(.+)/(.+)%.avi' )
			local cid = cname2id[ cname ]
			return vpath, cid
		end
	end
	local vid2path, vid2cid, vid2numf = {  }, {  }, {  }
	local lines = readLinesFrom( paths.concat( labelDir, labelFname ) )
	for vid, str in pairs( lines ) do
		local vpath, cid = parse( str )
		if vpath:match( 'v_HandStandPushups' ) then -- wtf?
			vpath = vpath:gsub( 'v_HandStandPushups', 'v_HandstandPushups' )
		end
		vpath = paths.concat( videoDir, vpath )
		vid2path[ vid ] = vpath
		vid2cid[ vid ] = tonumber( cid )
		local numf = 0
		for f in paths.files( vpath ) do numf = numf + 1 end
		vid2numf[ vid ] = numf - 2
		if vid % 100 == 0 then print( vid .. ' videos found for ' .. setName .. '.' ) end
		assert( vpath:len(  ) > 0 and tonumber( cid ) > 0 and numf > 2 )
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
