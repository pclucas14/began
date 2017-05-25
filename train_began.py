from model import * 
from DataHandler import * 
from ExpHandler import * 
from collections import OrderedDict 

# hyperparameters / parameters
params = OrderedDict()
params['batch_size'] = 64
params['tensor_shape'] = (64, 3, 64, 64)
params['initial_eta'] = 1e-4
params['lambda_k'] = 0.001
params['gamma'] = 0.7
params['load_weights'] = None #(12, 1000)#None# version/epoch tupple pair
params['optimizer'] = 'adam'
params['image_prepro'] = 'DCGAN' # (/250.; -0.5; /0.5) taken from DCGAN repo.
params['loss_comments'] = 'BEGAN loss with DCGAN architecture'
params['epoch_iter'] = 50
params['test'] = True

generator_layers = generator()
encoder_layers = encoder()
decoder_layers = generator()
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
loss_real = lasagne.objectives.squared_error(real_rec_imgs, images).mean()

# fake "loss"
fake_imgs = ll.get_output(generator) 
fake_enc_imgs = ll.get_output(encoder, inputs=fake_imgs)
fake_rec_imgs = ll.get_output(decoder, inputs=fake_enc_imgs)
loss_fake = lasagne.objectives.squared_error(fake_rec_imgs, fake_imgs).mean()

# generator/discriminator losses
gen_loss = loss_fake
disc_loss = loss_real - k * loss_fake
total_loss = gen_loss + disc_loss
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
M_global = loss_real + T.abs_(gamma*loss_real - loss_fake)

# function outputs
fn_output = OrderedDict()
fn_output['loss_real_mean'] = loss_real
fn_output['loss_fake_mean'] = loss_fake
fn_output['k'] = k
fn_output['gen_grads_norm'] = gen_grads_norm
fn_output['disc_grads_norm'] = disc_grads_norm
fn_output['M_global'] = M_global

eh.add_model('gen', generator_layers, fn_output)
eh.add_model('dec', decoder_layers, fn_output)
eh.add_model('enc', encoder_layers, fn_output)

# functions
print 'compiling functions'
train_model = theano.function(inputs=[index, k], 
                              outputs=fn_output.values(), 
                              updates=all_updates, 
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
        batch_no = dh.get_next_batch_no()
        model_out = train_model(batch_no, k_value)
        loss_real_mean, loss_fake_mean, k_value, _, _, _ = model_out

        # perform [hyper]parameter updates
        k_new = k_value + lambda_k * (gamma*loss_real_mean - loss_fake_mean)
        k_value = np.float32(np.clip(k_new, 0,1))

        error += np.array(model_out)
        eh.record('gen', np.array(model_out))

    # test out model
    batch_no = dh.get_next_batch_no()
    eh.save_image(test_gen()[0])
    eh.end_of_epoch()
    
    print epoch
    print("model  loss:\t\t{}".format(error / params['epoch_iter']))

 
