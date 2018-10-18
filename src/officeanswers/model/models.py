"""An implementation of DSSM, Deep Structured Semantic Model."""
import tensorflow as tf
import tensorflow_hub as hub

from keras.models import Model
from keras.layers import Dot
from keras.layers import Input
from keras.layers import Lambda
from matchzoo import engine

MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
EMBED_SIZE = (512,)


class UEModel(engine.BaseModel):
    def __init__(self):
        super().__init__()
        self._embed = hub.Module(MODULE_URL, trainable=True)

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['input_shapes'] = [(1,), (1,)]
        return params

    def build(self):
        """
        Build model structure.

        DSSM use Siamese arthitecture.
        """
        input_left = Input(name="text_left",
                           shape=(1,),
                           dtype=tf.string)
        input_right = Input(name="text_right",
                            shape=(1,),
                            dtype=tf.string)
        embed_left = Lambda(self._embedding_layer,
                            output_shape=EMBED_SIZE)(input_left)
        embed_right = Lambda(self._embedding_layer,
                             output_shape=EMBED_SIZE)(input_right)
        # Left input and right input.
        # Process left & right input.
        x = [embed_left,
             embed_right]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)

    def _embedding_layer(self, x):
        return self._embed(tf.squeeze(tf.cast(x, tf.string)))
