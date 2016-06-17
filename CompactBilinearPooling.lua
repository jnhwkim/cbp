local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

require 'spectralnet'

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
   -- self.output = torch.Tensor()
   self.gradInput = {}
   self.tmp = torch.Tensor()
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

function CompactBilinearPooling:conv(res,x,y)
   local batchSize = x:size(1)
   local dim = x:size(2)
   local function makeComplex(x,y)
      self.x_ = self.x_ or torch.CudaTensor(x:size(1),1,1,x:size(2),2):zero()
      self.x_[{{},{1},{1},{},{1}}]:copy(x)
      self.y_ = self.y_ or torch.CudaTensor(y:size(1),1,1,y:size(2),2):zero()
      self.y_[{{},{1},{1},{},{1}}]:copy(y)
   end
   makeComplex(x,y)
   self.fft_x = self.fft_x or torch.CudaTensor(batchSize,1,1,dim,2)
   self.fft_y = self.fft_y or torch.CudaTensor(batchSize,1,1,dim,2)
   res = res or torch.CudaTensor()
   res:resize(batchSize,1,1,dim*2)
   cufft.fft1d(self.x_:view(x:size(1),1,1,-1), self.fft_x)
   cufft.fft1d(self.y_:view(y:size(1),1,1,-1), self.fft_y)
   cufft.ifft1d(self.fft_x:cmul(self.fft_y), res)
   return res:view(batchSize,1,1,dim,2)[{{},{1},{1},{1}}]:squeeze()
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
   self:conv(self.output, self.y[1], self.y[2])
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
      self.tmp = self.tmp or gradOutput.new()
      self.tmp:resizeAs(gradOutput)

      if 1==k then
         self.tmp = self:conv(self.tmp, self.y[k%2+1], gradOutput)
      else
         self.tmp = self:conv(self.tmp, gradOutput, self.y[k%2+1])
      end
      self.gradInput[k]:index(self.tmp, 2, self.h[k])
      self.gradInput[k]:cmul(self.s[k]:repeatTensor(batchSize,1))
   end

   return self.gradInput
end