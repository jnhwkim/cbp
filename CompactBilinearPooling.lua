local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize, homogeneous)
   assert(outputSize and outputSize >= 1, 'missing outputSize!')
   self.outputSize = outputSize
   self.homogeneous = homogeneous
   self:reset()
end

function CompactBilinearPooling:reset()
   self.h = torch.Tensor()
   self.s = torch.Tensor()
   self.y = torch.Tensor()
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.tmp = torch.Tensor()
   self._tmp = torch.Tensor()
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
   end
end

function CompactBilinearPooling:conv(x, y)
   self.output:resizeAs(x):zero()
   if 1 == #x:size() then
      assert('not implemented')
   elseif 2 == #x:size() then
      assert(x:size(1)==y:size(1), 'should the same batch size')
      assert(x:size(2)==y:size(2), 'should the same dim size')
      for j=1,x:size(2) do  -- in
         local tmp = torch.cmul(x[{{},{j,x:size(2)}}], y[{{},{1,x:size(2)-j+1}}])
         self.output[{{},{j}}]:add(tmp:sum(2))
      end
      for j=1,x:size(2)-1 do  -- out
         local tmp = torch.cmul(x[{{},{1,j}}], y[{{},{y:size(2)-j+1,y:size(2)}}])
         self.output[{{},{j+1}}]:add(tmp:sum(2))
      end
   end
   return self.output
end

function CompactBilinearPooling:elt(x, y, offset)
   assert(x:size(1)==y:size(1), 'should the same batch size')
   assert(x:size(2)==y:size(2), 'should the same dim size')
   self._tmp:resize(x:size(1))
   local dim = x:size(2)
   local _tmp = torch.cmul(x[{{},{dim-offset+1,dim}}], y[{{},{1,offset}}])
   self._tmp:copy(_tmp:sum(2))
   if dim ~= offset then
      _tmp = torch.cmul(x[{{},{1,dim-offset}}], y[{{},{offset+1,dim}}])
      self._tmp:add(_tmp:sum(2))
   end
   return self._tmp
end

function CompactBilinearPooling:updateOutput(input)
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
   self:psi()
   self:conv(self.y[1], self.y[2])

   return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)
   local dim = input[1]:size(2)
   local batchSize = input[1]:size(1)
   self.gradInput:resizeAs(input[1]):zero()
   self.tmp:resizeAs(self.gradInput):zero()

   for k=1,2 do
      for i=1,dim do
         self.tmp[{{},{i}}]:add(self:elt(self.y[k%2+1],gradOutput,self.h[k][i]))
      end
      self.tmp:cmul(self.s[k]:repeatTensor(batchSize,1))
      self.gradInput:add(self.tmp)
   end

   return self.gradInput
end