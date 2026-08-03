# this file shows the training section using SGD optimizer
if train:
        for l in range(loop):
          if not who:
            # backward pass
            for p in x.parameters:
              p.grad=None
            x.loss.backward()
            for p in o.parameters:
              p.grad=None
            o.loss.backward()
    
            # optimizing...
            for p in x.parameters:
              p.data += (lr * p.grad)
            for p in o.parameters:
              p.data += (lr * p.grad)
          if who == 'x':
            for p in x.parameters:
              p.grad=None
            x.loss.backward()
            # optimizing...
            for p in x.parameters:
              p.data += (lr * p.grad)
          if who == 'o':
            for p in o.parameters:
              p.grad=None
            o.loss.backward()
            # optimizing...
            for p in o.parameters:
              p.data += (lr * p.grad)
