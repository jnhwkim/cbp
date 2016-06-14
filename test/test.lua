require 'nn'
require 'cunn'

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

function cbptest.testConv()
   local x = torch.CudaTensor{{1,2,3},{2,3,4}}
   local y = torch.CudaTensor{{1,1,1},{2,2,2}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{6,6,6},{18,18,18}}
   c:conv(x,y)
   ans:add(-c.output)
   assert(ans:norm() < precision, ans:norm())

   local x = torch.CudaTensor{{1,2,3,1,1},{2,3,4,1,1}}
   local y = torch.CudaTensor{{1,1,1,1,1},{2,2,2,1,1}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{8,8,8,8,8},{20,19,17,15,17}}
   c:conv(x,y)
   ans:add(-c.output)
   assert(ans:norm() < precision, ans:norm())
end

function cbptest.testGrad()
   assert(true)
end

function cbp.test(tests)
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end