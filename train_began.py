from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['tensor_shape'] = (64, 3, 64, 64)
params['initial_eta'] = 2e-4
params['lambda_k'] = 0.001
params['gamma'] = 0.5
params['load_weights'] = None#(9,200) #(12, 1000)#None# version/epoch tupple pair
params['optimizer'] = 'adam'
params['image_prepro'] = 'DCGAN' # (/250.; -0.5; /0.5) taken from DCGAN repo.
params['loss_comments'] = 'BEGAN loss with DCGAN architecture'
params['epoch_iter'] = 40
params['gen_iter'] = 1
params['disc_iter'] = 1
params['test'] = True

generator_layers = generator_DCGAN(batch_size=params['batch_size'])
encoder_layers = encoder_DCGAN(batch_size=params['batch_size'])
decoder_layers = generator_DCGAN(batch_size=params['batch_size'])
generator = generator_layers[-1]
encoder = encoder_layers[-1] # discriminator
decoder = decoder_layers[-1] # discriminator

dh = DataHandler(tensor_shape=params['tensor_shape'])
eh = ExpHandler(params, test=params['test'])

# placeholders 
images = T.tensor4('images from dataset')
index = T.lscalar() # index to a [mini]batch
eta = theano.shared(lasagne.utils.floatX(params['initial_eta']))
k = T.fscalar()
k_value = np.float32(0.)
M_global = T.fscalar()
lambda_k = params['lambda_k']
gamma = params['gamma']

# params
gen_params = ll.get_all_params(generator, trainable=True)
disc_params = ll.get_all_params([encoder, decoder], trainable=True)

# real "loss" 
real_enc_imgs = ll.get_output(encoder, inputs=images) # encoding of real images
real_rec_imgs = ll.get_output(decoder, inputs=real_enc_imgs) # reconstruction of real images
loss_real = lasagne.objectives.squared_error(real_rec_imgs, images)
loss_real_mean = loss_real.mean()
#import pdb; pdb.set_trace()
loss_real_std = T.std(loss_real, axis=0)

# fake "loss"
fake_imgs = ll.get_output(generator) 
fake_enc_imgs = ll.get_output(encoder, inputs=fake_imgs)
fake_rec_imgs = ll.get_output(decoder, inputs=fake_enc_imgs)
loss_fake = lasagne.objectives.squared_error(fake_rec_imgs, fake_imgs)
loss_fake_mean = loss_fake.mean()
loss_fake_std = T.std(loss_fake, axis=0)

# generator/discriminator losses
# began loss
std_penalty = lasagne.objectives.squared_error(( 1/(loss_fake_std+1e-8)), (1/(loss_real_std+1e-8))).mean()
gen_loss = loss_fake_mean + 0.01 * std_penalty
disc_loss = loss_real_mean - k * loss_fake_mean
total_loss = gen_loss + disc_loss

'''
# trying something
gen_loss = T.abs_((loss_real/loss_fake) - 1).mean()
disc_loss = T.abs_((loss_real/loss_fake) - 0).mean()
'''
gen_grads = theano.grad(gen_loss, wrt=gen_params)
disc_grads = theano.grad(disc_loss, wrt=disc_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
disc_grads_norm = sum(T.sum(T.square(grad)) for grad in disc_grads) / len(disc_grads)

# updates
updates_gen = optimizer_factory(params['optimizer'], gen_grads, gen_params, eta)
updates_disc = optimizer_factory(params['optimizer'], disc_grads, disc_params, eta)
all_updates = updates_gen.copy()
all_updates.update(updates_disc)

# setting BEGAN parameters
M_global = loss_real_mean + T.abs_(gamma*loss_real_mean - loss_fake_mean)

# function outputs
fn_output = OrderedDict()
fn_output['loss_real_mean'] = loss_real_mean
fn_output['loss_fake_mean'] = loss_fake_mean
fn_output['loss_real_std'] = loss_real_std.mean()
fn_output['loss_fake_std'] = loss_fake_std.mean()
fn_output['std_penalty'] = std_penalty
fn_output['k'] = k
fn_output['gen_grads_norm'] = gen_grads_norm
fn_output['disc_grads_norm'] = disc_grads_norm
fn_output['M_global'] = M_global

eh.add_model('gen', generator_layers, fn_output)
eh.add_model('dec', decoder_layers, fn_output)
eh.add_model('enc', encoder_layers, fn_output)

# functions
print 'compiling functions'
train_gen = theano.function(inputs=[index, k], 
                              outputs=fn_output.values(), 
                              updates=updates_gen, 
                              givens={images: dh.GPU_image[index * params['batch_size']: 
                                                       (index+1) * params['batch_size']]},
                              name='train_model')

train_disc = theano.function(inputs=[index, k], 
                              outputs=fn_output.values(), 
                              updates=updates_disc, 
                              givens={images: dh.GPU_image[index * params['batch_size']: 
                                                       (index+1) * params['batch_size']]},
                              name='train_model')

test_gen =     theano.function(inputs=[],
                               outputs=[fake_imgs],
                               name='test_gen')

'''
training section 
'''
print 'staring training'
for epoch in range(3000000):
    error = 0

    for _ in range(params['epoch_iter']):
        for _ in range(params['gen_iter']): 
            batch_no = dh.get_next_batch_no()
            model_out = train_gen(batch_no, k_value)
            loss_real_mean, loss_fake_mean, _, _, _, k_value, _, _, _ = model_out

            # perform [hyper]parameter updates
            k_new = k_value + lambda_k * (gamma*loss_real_mean - loss_fake_mean)
            k_value = np.float32(np.clip(k_new, 0,1))
            # import pdb; pdb.set_trace()
            error += np.array(model_out)
            eh.record('gen', np.array(model_out))

        for _ in range(params['disc_iter']):
            batch_no = dh.get_next_batch_no()
            model_out = train_disc(batch_no, k_value)
            loss_real_mean, loss_fake_mean, _, _, _, k_value, _, _, _= model_out

            # perform [hyper]parameter updates
            k_new = k_value + lambda_k * (gamma*loss_real_mean - loss_fake_mean)
            k_value = np.float32(np.clip(k_new, 0,1))

            error += np.array(model_out)
            eh.record('dec', np.array(model_out))

    # test out model
    batch_no = dh.get_next_batch_no()
    eh.save_image(test_gen()[0])
    eh.end_of_epoch()
    
    print epoch
    print("model  loss:\t\t{}".format(error / (params['gen_iter']*params['epoch_iter'])))

 
