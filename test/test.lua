require 'nn'

local cbptest = torch.TestSuite()
local precision = 1e-5

function cbptest.testPsi()
   assert(true, 'Example')
end

function cbp.test(tests)
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end