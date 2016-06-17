require 'nn'
require 'cunn'

local cbptest = torch.TestSuite()
local precision = 1e-5

function cbptest.testPsi()
   local homogeneous = true
   local batch = 1
   local dim = 5
   local S = 100
   local x = torch.rand(batch,dim):cuda()
   local y = torch.rand(batch,dim):cuda()
   local tmp = 0
   local c
   for s=1,S do
      c = nn.CompactBilinearPooling(dim, homogeneous):cuda()
      c:forward({x,y})
      tmp = tmp + c.y[1][1]:dot(c.y[2][1])
   end
   local xy = x[1]:dot(y[1])
   local diff = math.abs(tmp/S - xy)
   assert(diff / xy < .1, 'error_ratio='..diff / xy..', E[<phi(x,h,s),phi(y,h,s)>]=<x,y>')
end

function cbptest.testConv()
   local x = torch.CudaTensor{{1,2,3},{2,3,4}}
   local y = torch.CudaTensor{{1,1,1},{2,2,2}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{6,6,6},{18,18,18}}
   local output = torch.CudaTensor()
   output = c:conv(x,y)  -- cuda only
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())

   local x = torch.CudaTensor{{1,2,3,1,1},{2,3,4,1,1}}
   local y = torch.CudaTensor{{1,1,1,1,1},{2,2,2,1,1}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{8,8,8,8,8},{17.5,17.5,17.75,17.75,17.5}}
   output = c:conv(x,y)
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())
end

function cbptest.testGrad()
   -- local x = torch.Tensor{{1,2,3},{2,3,4}}
   -- local y = torch.Tensor{{1,1,1},{2,2,2}}
   -- local c = nn.CompactBilinearPooling(x:size(2))
   -- local diff = torch.Tensor{{6,6,6},{18,18,18}}
   -- c:forward{x:add(torch.rand(2,3):mul(precision)),y:add(torch.rand(2,3):mul(precision))}
   -- diff:add(-c.output):div(precision)
   -- c:backward({x:add(torch.rand(2,3):mul(precision)),y:add(torch.rand(2,3):mul(precision))}, torch.Tensor(2,3):fill(1))
   -- local grad = c.gradInput
   -- local ans = diff:add(-grad)
   -- assert(ans:norm() < precision, ans:norm())
end

function cbp.test(tests)
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end