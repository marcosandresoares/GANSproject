require 'image'
require 'nn'

local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 32,
    noisetype = 'normal',
    net = '',
    imsize = 1,
    noisemode = 'random',
    name = 'generation1',
    gpu = 1,
    display = 1,
    nz = 100
}

for k,v in pairs(opt) do
    opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k]
end
print(opt)
if opt.display == 0 then
    opt.display = false
end

assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = torch.load(opt.net)

--[[
    For older models, there was nn.View on top 
    which is unnecessary and hinders convolutional generations
]]--
if torch.type(net:get(1)) == 'nn.View' then
    net.remove(1)
end

print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1,1)
elseif opt.noisetype == 'normal' then
    noise:normal(0,1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1,1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1,1)

if opt.noisemode == 'line' then
    line = torch.linspace(0, 1, batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noise == 'linefull1d' then
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize then
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
    assert(opt.batchSize == 1), 'for linefull mode, give batchSize(1) and imsize > 1'
    line = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end

local sample_input = torch.randn(2, 100, 1, 1)
if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    sample_input = sample_input:cuda()
else
    sample_input = sample_input:float()
    net:float()
end

--[[
    A function to setup double buffering across the network
    This drastically reduces the memory needed to generate samples
]]--
optnet.optimizeMemory(net, sample_input)

local images  net:forward(noise)
print('Images size: ', images:size(1)..'x'..images:size(2)..'x'..images:size(3)..'x'..images:size(4))
images:add(1):mul(0.5)
print('Min : ' , images:min())
print('Max : ' , images:max())
print('Mean : ' , images:mean())
print('Standard Deviation : ' , images:std())
image.save(opt.name..'.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed Image')
end