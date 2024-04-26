import torch as tc

class accuracy:
    def __call__(self, outputs: tc.Tensor, labels: tc.Tensor) -> float:
        """Computes the accuracy
        Args:
            outputs: tensor of size (N, K) where N is the batch size and K is number of classes
            labels: tensor of size (N, 1) where N is batch size
        """
        values = tc.argmax(outputs, dim = 1)
        bool_mask = (values == labels).int()
        self.score =  bool_mask.sum()/len(bool_mask) * 100
        return self.score
        
    
