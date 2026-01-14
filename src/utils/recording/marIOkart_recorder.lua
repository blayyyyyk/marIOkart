-- This Lua script works in both BizHawk and DeSmuME. For DeSmuME, you must download additional files for lua.
-- You need lua51.dll and lua5.1.dll, both of which can be obtained at the following location:
-- https://sourceforge.net/projects/luabinaries/files/5.1.5/Tools%20Executables/lua-5.1.5_Win64_bin.zip/download

-- Before starting, make sure we are not about to overwrite data.
local fs = io.open("completed_race_1.json")
if fs ~= nil then
	io.close(fs)
	error("File completed_race_1.json already exists. Please move or delete prior recordings.")
end

-----------------------------------
-- JSON
-----------------------------------
json = {}

local function __json__()
	local function kind_of(obj)
		if type(obj) ~= 'table' then return type(obj) end
		local i = 1
		for _ in pairs(obj) do
		if obj[i] ~= nil then i = i + 1 else return 'table' end
		end
		if i == 1 then return 'table' else return 'array' end
	end

	local function escape_str(s)
		local in_char	= {'\\', '"', '/', '\b', '\f', '\n', '\r', '\t'}
		local out_char = {'\\', '"', '/',	'b',	'f',	'n',	'r',	't'}
		for i, c in ipairs(in_char) do
		s = s:gsub(c, '\\' .. out_char[i])
		end
		return s
	end

	-- Returns pos, did_find; there are two cases:
	-- 1. Delimiter found: pos = pos after leading space + delim; did_find = true.
	-- 2. Delimiter not found: pos = pos after leading space;		 did_find = false.
	-- This throws an error if err_if_missing is true and the delim is not found.
	local function skip_delim(str, pos, delim, err_if_missing)
		pos = pos + #str:match('^%s*', pos)
		if str:sub(pos, pos) ~= delim then
		if err_if_missing then
			error('Expected ' .. delim .. ' near position ' .. pos)
		end
		return pos, false
		end
		return pos + 1, true
	end

	-- Expects the given pos to be the first character after the opening quote.
	-- Returns val, pos; the returned pos is after the closing quote character.
	local function parse_str_val(str, pos, val)
		val = val or ''
		local early_end_error = 'End of input found while parsing string.'
		if pos > #str then error(early_end_error) end
		local c = str:sub(pos, pos)
		if c == '"'	then return val, pos + 1 end
		if c ~= '\\' then return parse_str_val(str, pos + 1, val .. c) end
		-- We must have a \ character.
		local esc_map = {b = '\b', f = '\f', n = '\n', r = '\r', t = '\t'}
		local nextc = str:sub(pos + 1, pos + 1)
		if not nextc then error(early_end_error) end
		return parse_str_val(str, pos + 2, val .. (esc_map[nextc] or nextc))
	end

	-- Returns val, pos; the returned pos is after the number's final character.
	local function parse_num_val(str, pos)
		local num_str = str:match('^-?%d+%.?%d*[eE]?[+-]?%d*', pos)
		local val = tonumber(num_str)
		if not val then error('Error parsing number at position ' .. pos .. '.') end
		return val, pos + #num_str
	end
	
	json.stringify = function(obj, as_key)
		local s = {} -- We'll build the string as an array of strings to be concatenated.
		local kind = kind_of(obj) -- This is 'array' if it's an array or type(obj) otherwise.
		if kind == 'array' then
			if as_key then error('Can\'t encode array as key.') end
			s[#s + 1] = '['
			for i, val in ipairs(obj) do
				if i > 1 then s[#s + 1] = ', ' end
				s[#s + 1] = json.stringify(val)
			end
			s[#s + 1] = ']'
		elseif kind == 'table' then
			if as_key then error('Can\'t encode table as key.') end
			s[#s + 1] = '{'
			for k, v in pairs(obj) do
				if #s > 1 then s[#s + 1] = ', ' end
				s[#s + 1] = json.stringify(k, true)
				s[#s + 1] = ':'
				s[#s + 1] = json.stringify(v)
			end
			s[#s + 1] = '}'
		elseif kind == 'string' then
			return '"' .. escape_str(obj) .. '"'
		elseif kind == 'number' then
			if as_key then return '"' .. tostring(obj) .. '"' end
			return tostring(obj)
		elseif kind == 'boolean' then
			return tostring(obj)
		elseif kind == 'nil' then
			return 'null'
		else
			error('Unjsonifiable type: ' .. kind .. '.')
		end
		return table.concat(s)
	end
	
	-- This is a one-off table to represent the null value.
	json.null = {}
	
	json.parse = function(str, pos, end_delim)
		pos = pos or 1
		if pos > #str then error('Reached unexpected end of input.') end
		local pos = pos + #str:match('^%s*', pos)	-- Skip whitespace.
		local first = str:sub(pos, pos)
		if first == '{' then	-- Parse an object.
			local obj, key, delim_found = {}, true, true
			pos = pos + 1
			while true do
				key, pos = json.parse(str, pos, '}')
				if key == nil then return obj, pos end
				if not delim_found then error('Comma missing between object items.') end
				pos = skip_delim(str, pos, ':', true)	-- true -> error if missing.
				obj[key], pos = json.parse(str, pos)
				pos, delim_found = skip_delim(str, pos, ',')
			end
		elseif first == '[' then	-- Parse an array.
			local arr, val, delim_found = {}, true, true
			pos = pos + 1
			while true do
				val, pos = json.parse(str, pos, ']')
				if val == nil then return arr, pos end
				if not delim_found then error('Comma missing between array items.') end
				arr[#arr + 1] = val
				pos, delim_found = skip_delim(str, pos, ',')
			end
		elseif first == '"' then	-- Parse a string.
			local raw, nextPos = parse_str_val(str, pos + 1)
		if #raw == 9 and raw:sub(1, 1) == "#" then
			-- JSON does not support hex. This is a support of hex.
			-- I chose to do it this way only because a json file I was given has hex values formatted this way.
			return tonumber(raw:sub(2), 16), nextPos
		else
			return raw, nextPos
		end
		elseif first == '-' or first:match('%d') then	-- Parse a number.
			return parse_num_val(str, pos)
		elseif first == end_delim then	-- End of an object or array.
			return nil, pos + 1
		else	-- Parse true, false, or null.
			local literals = {['true'] = true, ['false'] = false, ['null'] = json.null}
			for lit_str, lit_val in pairs(literals) do
				local lit_end = pos + #lit_str - 1
				if str:sub(pos, lit_end) == lit_str then return lit_val, lit_end + 1 end
			end
			local pos_info_str = 'position ' .. pos .. ': ' .. str:sub(pos, pos + 10)
			error('Invalid json syntax starting at ' .. pos_info_str)
		end
	end
end
__json__()
--------------------

-----------------------------------
-- DeSmuME compatibility
-----------------------------------
local is_desmume = bizstring == nil
if is_desmume then
	memory.read_u8 = memory.readbyte
	memory.read_u16_le = memory.readword
	memory.read_u32_le = memory.readdword
	memory.read_s32_le = memory.readdwordsigned
	memory.read_bytes_as_array = memory.readbyterange
	
	event.onexit = emu.registerexit
end
-----------------------------------


local somePointerWithRegionAgnosticAddress = memory.read_u32_le(0x2000B54)
local valueForUSVersion = 0x0216F320
local ptrOffset = somePointerWithRegionAgnosticAddress - valueForUSVersion

local raceConfig = memory.read_u32_le(0x021759a0 + ptrOffset)
local playerConfig = raceConfig + 0x68
local ptr = memory.read_u32_le(0x021755FC + ptrOffset)
local playerTimes = ptr + 0x20

local inputUnits = 0x02175608 + ptrOffset
local racerDataPtr = 0x0217ACF8 + ptrOffset
local sceneStatePtr = 0x021759AC + ptrOffset
local raceStatePtr = 0x0217AA34 + ptrOffset
local raceStatusPtr = 0x021755FC + ptrOffset

local function FramesSinceRaceStart()
	-- Check if racer exists.
	local currentRacersPtr = memory.read_s32_le(racerDataPtr)
	if currentRacersPtr == 0 then
		return -1
	end

	-- Race won't begin until scene is running.
	local scenePtr = memory.read_u32_le(sceneStatePtr)
	local sceneState = memory.read_u32_le(scenePtr + 0x10c)
	if sceneState < 2 or sceneState > 3 then
		return -1
	end
	
	-- Check if race has begun. (This static pointer points to junk on the main menu, which is why we checked racer data first.)
	return memory.read_s32_le(memory.read_s32_le(raceStatePtr) + 4)
end

local function collectData(fileName)
	memory.usememorydomain("ARM9 System Bus")
	local data = {
		characterId = memory.read_u8(playerConfig + 0x00),
		kartId = memory.read_u8(playerConfig + 0x04),
		courseId = memory.read_u8(raceConfig),
		ghostInputs = memory.read_bytes_as_array(memory.read_u32_le(inputUnits + 0x28), 0xdcc),
	}

	fileName = fileName .. ".json"
	local fs = io.open(fileName, "w")
	if fs == nil then error("could not open/create " .. fileName) end
	fs:write(json.stringify(data))
	io.close(fs)
	
	print("Wrote data file " .. fileName)
end

local finished = false
local lastFinishId = 0
local lastQuitId = 0

-- Let the user know we are running
print("Recording data for marIOkart.")
event.onexit(function() print("marIOkart: Stopped.") end)

local function main()
	local frames = FramesSinceRaceStart()
	if frames >= 0 then
		local raceStatus = memory.read_u32_le(raceStatusPtr)
		local raceProgress = memory.read_s32_le(raceStatus + 0x14 + 0x44)
		if raceProgress >= 0x1000 and not finished then
			-- Finished the race (TODO: Does this work when no ghost will be saved?)
			finished = true
			lastFinishId = lastFinishId + 1
			collectData("completed_race_" .. lastFinishId)
		end
		
		-- Detect quitting.
		local scenePtr = memory.read_u32_le(sceneStatePtr)
		local sceneState = memory.read_u32_le(scenePtr + 0x10c)
		if sceneState == 3 and not finished then
			finished = true
			lastQuitId = lastQuitId + 1
			local inputRecAddr = inputUnits + 0x28
			local recordingAddr = memory.read_u32_le(inputRecAddr)
			memory.write_u16_le(recordingAddr, memory.read_u16_le(inputRecAddr + 4) * 2)
			collectData("incomplete_attempt_" .. lastQuitId)
		end
	else
		finished = false
	end
end

if is_desmume then
	emu.registerafter(main)
else
	while true do
		main()
		emu.frameadvance()
	end
end
