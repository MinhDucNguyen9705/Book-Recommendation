import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def init_normal():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    
class GMF(layers.Layer):

    def __init__(self, num_users, num_items, latent_dim, regs=[0, 0], **kwargs):
        super(GMF, self).__init__(**kwargs)
        
        self.user_embedding = layers.Embedding(
                                                input_dim=num_users, 
                                                output_dim=latent_dim,
                                                name='user_embedding', 
                                                embeddings_initializer = init_normal(), 
                                                embeddings_regularizer = keras.regularizers.l2(regs[0])
                                            )
        self.item_embedding = layers.Embedding(
                                                input_dim=num_items, 
                                                output_dim=latent_dim,
                                                name='item_embedding', 
                                                embeddings_initializer = init_normal(), 
                                                embeddings_regularizer = keras.regularizers.l2(regs[1])
                                            )
        self.user_latent = layers.Flatten()
        self.item_latent = layers.Flatten()
    
    def call(self, inputs):
        user_input, item_input = inputs
        
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        
        user_latent = self.user_latent(user_embedding)
        item_latent = self.item_latent(item_embedding)

        output = layers.Multiply()([user_latent, item_latent])

        return output
        
class MLP(layers.Layer):

    def __init__(self, num_users, num_items, num_genres, layers_=[20,10], regs=[0, 0, 0], **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_layers = len(layers_)

        self.user_embedding = layers.Embedding(
                                                input_dim=num_users, 
                                                output_dim=layers_[0]//2,
                                                name='user_embedding', 
                                                embeddings_initializer = init_normal(), 
                                                embeddings_regularizer = keras.regularizers.l2(regs[0])
                                            )
        self.item_embedding = layers.Embedding(
                                                input_dim=num_items,
                                                output_dim=layers_[0]//2,
                                                name='item_embedding', 
                                                embeddings_initializer = init_normal(), 
                                                embeddings_regularizer = keras.regularizers.l2(regs[1])
                                            )
        self.genre_embedding = layers.Embedding(
                                                input_dim=num_genres,
                                                output_dim=layers_[0]//2, 
                                                name='genre_embedding',
                                                embeddings_initializer = init_normal(), 
                                                embeddings_regularizer = keras.regularizers.l2(regs[2])
                                            )
        self.user_latent = layers.Flatten()
        self.item_latent = layers.Flatten()
        self.genre_latent = layers.Flatten()
    
        self.fcn = keras.Sequential([layers.Dense(layers_[i], activation='relu') for i in range (self.num_layers)])
                
    def call(self, inputs):
        user_input, item_input, genre_input = inputs
        
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        genre_embedding = self.genre_embedding(genre_input)
        
        user_latent = self.user_latent(user_embedding)
        item_latent = self.item_latent(item_embedding)
        genre_latent = self.genre_latent(genre_embedding)

        concat = layers.Concatenate()([user_latent, item_latent, genre_latent])
        output = self.fcn(concat)

        return output
        
def NeuMF(num_users, num_items, num_genres, latent_dim=10, layers_=[10], regs=[0, 0, 0, 0, 0]):
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')
    genre_input = layers.Input(shape=(num_genres,), dtype='float32', name='genre_input')

    mf_vector = GMF(num_users, num_items, latent_dim, regs[:2], name='GMF')([user_input, item_input])
    mlp_vector = MLP(num_users, num_items, num_genres, layers_, regs[2:], name='MLP')([user_input, item_input, genre_input])

    predict_vector = layers.Concatenate(name='concatenate')([mf_vector, mlp_vector])
    # predict_vector = mlp_vector
    prediction = layers.Dense(1, activation='sigmoid', name='output')(predict_vector)*4+1
    # prediction = layers.Dense(1)(predict_vector)
    # prediction = layers.Lambda(lambda x: tf.clip_by_value(x, 1.0, 5.0))(prediction)
    
    model = keras.Model(inputs=[user_input, item_input, genre_input], outputs=prediction)

    return model