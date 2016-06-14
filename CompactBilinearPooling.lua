local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

signal = require 'signal'

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize, homogeneous)
   assert(outputSize and outputSize >= 1, 'missing outputSize!')
   self.outputSize = outputSize
   self.homogeneous = homogeneous
   self:reset()
   self.debug = false
end

function CompactBilinearPooling:reset()
   self.h = torch.Tensor()
   self.s = torch.Tensor()
   self.y = torch.Tensor()
   self.output = torch.Tensor()
   self.gradInput = {}
   self.tmp = torch.Tensor()
   self._tmp = torch.Tensor()
end

function CompactBilinearPooling:sample()
   self.h:uniform(0,self.outputSize):ceil()
   self.s:uniform(0,2):floor():mul(2):add(-1)
end

function CompactBilinearPooling:psi()
   self.y:zero()
   for i=1,2 do
      if self.homogeneous then  -- using the same samples
         for j=1,self.h[1]:size(#self.h[1]:size()) do
           local y_ = self.y[i][self.h[1][j]]
           self.y[i][self.h[1][j]] = y_ + self.s[1][j]*self.input[i][j]
        end
      else
         local batchSize = self.input[1]:size(1)
         self.y[i]:indexAdd(2,self.h[i],torch.cmul(self.s[i]:repeatTensor(batchSize,1),self.input[i]))
      end
   end
end

-- function CompactBilinearPooling:conv(x, y)
--    self.output:resizeAs(x):zero()
--    if 1 == #x:size() then
--       assert('not implemented')
--    elseif 2 == #x:size() then
--       assert(x:size(1)==y:size(1), 'should the same batch size')
--       assert(x:size(2)==y:size(2), 'should the same dim size')
--       for j=1,x:size(2) do  -- in
--          local tmp = torch.cmul(x[{{},{j,x:size(2)}}], y[{{},{1,x:size(2)-j+1}}])
--          self.output[{{},{j}}]:add(tmp:sum(2))
--          if j~=1 then
--             local tmp = torch.cmul(x[{{},{1,j-1}}], y[{{},{y:size(2)-j+2,y:size(2)}}])
--             self.output[{{},{j}}]:add(tmp:sum(2))
--          end
--       end
--    end
--    return self.output
-- end

function CompactBilinearPooling:conv(x, y)
   local batchSize = x:size(1)
   local dim = x:size(2)
   self.output:resizeAs(x)
   for i=1,batchSize do
      self.output[i]:copy(signal.ifft(signal.fft(x[i]:float()):cmul(signal.fft(y[i]:float())))[{{},{1}}])
   end
   return self.output
end

-- require 'spectralnet'
-- function CompactBilinearPooling:conv(x,y)
--    local batchSize = x:size(1)
--    local dim = x:size(2)
--    local function makeComplex(res, x)
--       res[{{},{1},{1},{},{1}}]:copy(x)
--       return res:resize(x:size(1),1,1,x:size(2)*2)
--    end
--    self.x_ = self.x_ or torch.CudaTensor(x:size(1),1,1,x:size(2),2):zero()
--    self.y_ = self.y_ or torch.CudaTensor(y:size(1),1,1,y:size(2),2):zero()
--    makeComplex(self.x_, x)
--    makeComplex(self.y_, y)
--    self.output:resize(batchSize,1,1,dim*2)
--    self.fft_x = self.fft_x or torch.CudaTensor(batchSize,1,1,dim,2)
--    self.fft_y = self.fft_y or torch.CudaTensor(batchSize,1,1,dim,2)
--    cufft.fft1d(self.x_, self.fft_x)
--    cufft.fft1d(self.y_, self.fft_y)
--    cufft.ifft1d(torch.cmul(self.fft_x,self.fft_y),self.output)
--    self.output=self.output:resize(batchSize,1,1,dim,2):select(2,1):select(2,1):select(3,1)
--    return self.output
-- end

function CompactBilinearPooling:elt(res, x, y, offset)
   assert(x:size(1)==y:size(1), 'should the same batch size')
   assert(x:size(2)==y:size(2), 'should the same dim size')
   local dim = x:size(2)
   local _tmp = torch.cmul(x[{{},{dim-offset+1,dim}}], y[{{},{1,offset}}]):sum(2)
   res[{{},{offset}}]:copy(_tmp)
   if dim ~= offset then
      _tmp = torch.cmul(x[{{},{1,dim-offset}}], y[{{},{offset+1,dim}}])
      res[{{},{offset}}]:add(_tmp:sum(2))
   end
end

function CompactBilinearPooling:updateOutput(input)
   if self.debug then sys.tic(1) end
   self.input = input
   local inputSizes1 = input[1]:size()
   local inputSizes2 = input[2]:size()
   local sizes1 = inputSizes1[#inputSizes1]
   local sizes2 = inputSizes2[#inputSizes2]
   self.h:resize(2, sizes1)
   self.s:resize(2, sizes1)
   self:sample()

   if 2 > #inputSizes1 then  -- no batch
      self.y:resize(2, self.outputSize)
   elseif 2 == #inputSizes1 then  -- batch
      local batchSize = inputSizes1[1]
      self.y:resize(2, batchSize, self.outputSize)
   else
      assert(false, '# of dimensions > 2')
   end
   if self.debug then print('pre:', sys.toc(1)) end
   self:psi()
   if self.debug then print('psi:', sys.toc(1)) end
   self:conv(self.y[1], self.y[2])
   if self.debug then print('conv:', sys.toc(1)) end

   return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)
   local dim = input[1]:size(2)
   local batchSize = input[1]:size(1)
   self.gradInput = self.gradInput or {}

   for k=1,2 do
      self.gradInput[k] = self.gradInput[k] or input[k].new()
      self.gradInput[k]:resizeAs(input[k]):zero()
      local tmp = gradOutput.new()
      tmp:resizeAs(gradOutput)
      -- for d=1,gradOutput:size(2) do
      --    self:elt(tmp,self.y[k%2+1],gradOutput,d)
      -- end
      for i=1,batchSize do
         tmp[i]:copy(signal.ifft(signal.fft(self.y[k%2+1][i]:float()):cmul(signal.fft(gradOutput[i]:float())))[{{},{1}}])
      end
      self.gradInput[k]:index(tmp, 2, self.h[k])
      self.gradInput[k]:cmul(self.s[k]:repeatTensor(batchSize,1))
   end

   return self.gradInput
end