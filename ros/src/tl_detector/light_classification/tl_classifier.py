from styx_msgs.msg import TrafficLight
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, is_site):
        self.model = None
        if is_site:
            # load model from Damian:
            model_path = 'SiteModel.pb' # TODO: change
        else:
            # load model from Marvin (?)
            model_path = 'SimModel.pb' # TODO: change

        self.model = load_graph(model_path)
        self.session = tf.Session(graph=self.model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        #return TrafficLight.UNKNOWN

        # TODO: what do the models return? do they return the probs / logits? 
        # or do they already select and return the highest probability state??  
        # or is this different for each of our models (i.e. for site and for sim)

        with self.model.as_default():
            with self.session as sess:
                # TODO: are these correct for our models? most prob not. getinfo from Damian
                # TODO: I assume it will be different for site and sim. 
                # then better to write two "get_classification" functions
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')


                (score, result) = sess.run([detect_scores, detect_classes], 
                                   feed_dict={image_tensor: image})

                if score > 0.7:   #TODO: tune this threshold
                    return result #TODO: make sure the enumeration used in models are
                                  # the same as in styx_msgs/TrafficLight
        
        # score was not high enough:
        return TrafficLight.UNKNOWN


    # CODE FROM https://github.com/alex-lechner/Traffic-Light-Classification
    def load_graph(self, graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph