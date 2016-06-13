require 'nn'

local cbptest = torch.TestSuite()
local precision = 1e-5

function cbptest.testPsi()
   local homogeneous = true
   local dim = 5
   local S = 1000
   local c = nn.CompactBilinearPooling(dim, homogeneous)
   local x = torch.rand(dim):cuda()
   local y = torch.rand(dim):cuda()
   local tmp = 0
   for s=1,S do
      c:forward({x,y})
      tmp = tmp + c.y[1]:dot(c.y[2])
   end
   local diff = math.abs(tmp/S - x*y)
   assert(diff / (x*y) < .1, 'error_ratio='..diff / (x*y)..', E[<phi(x,h,s),phi(y,h,s)>]=<x,y>')
end

function cbp.test(tests)
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end