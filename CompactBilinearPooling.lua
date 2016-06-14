local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize, homogeneous)
   self.outputSize = outputSize
   self.homogeneous = homogeneous
   self.tmp = torch.Tensor()
   self:reset()
end

function CompactBilinearPooling:reset()
   self.h = torch.LongTensor()
   self.s = torch.IntTensor()
   self.y = torch.Tensor()
   self.output = torch.Tensor()
end

function CompactBilinearPooling:sample()
   for i=1,2 do
      for k=1,self.h[i]:size(#self.h[i]:size()) do
         self.h[i][k] = torch.random(1,self.outputSize)  -- sample from (1,..,C)
         self.s[i][k] = torch.random(0,1)*2-1  -- sample from (-1,1)
      end
   end
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
         for j=1,self.h[i]:size(#self.h[i]:size()) do
            if 1 == #self.input[i]:size() then
               local y_ = self.y[i][self.h[i][j]]
               y_ = y_ + self.s[i][j]*self.input[i][j]
            elseif 2 == #self.input[i]:size() then
               local y_ = self.y[i][{{},{self.h[i][j]}}]
               y_ = y_:add(self.s[i][j]*self.input[i][{{},{j}}])
            end
         end
      end
      self.y:typeAs(self.input[i])
   end
end

function CompactBilinearPooling:conv(x, y)
   self.output = self.output:typeAs(x)
   self.output:resizeAs(x):zero()
   if 1 == #x:size() then
      assert('not implemented')
   elseif 2 == #x:size() then
      assert(x:size(1)==y:size(1), 'should the same batch size')
      assert(x:size(2)==y:size(2), 'should the same dim size')
      local str_idx = math.floor(x:size(2)/2) + 1
      for j=str_idx,x:size(2) do  -- first half
         local tmp = x[{{},{j,x:size(2)}}]:clone()
         tmp:cmul(y:narrow(2,j-str_idx+1,x:size(2)-j+1))
         self.output[{{},{j-str_idx+1}}]:add(tmp:sum(2))
      end
      local end_idx = math.floor(x:size(2)/2)
      for j=end_idx,x:size(2) do
         local tmp = x[{{},{1,j}}]:clone():typeAs(x)
         tmp:cmul(y[{{},{y:size(2)-j+1,y:size(2)}}])
         self.output[{{},{j-end_idx+1}}]:add(tmp:sum(2))
      end
   end
   return self.output
end

function CompactBilinearPooling:func(params, x, y)
   local input = {x,y}
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
   self.y=self.y:typeAs(input[1])
   self:psi()
   self:conv(self.y[1], self.y[2])
   return self.output
end

function CompactBilinearPooling:updateOutput(input)
   self.output = self:func(nil, input[1], input[2])
   return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)

end