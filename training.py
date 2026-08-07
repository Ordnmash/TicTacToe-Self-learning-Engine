# this file shows the training section using SGD optimizer
if train:
  for l in range(loop):
    if not who: # this trains both parts of the model.
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

    # with who passed as 'x' we only train specific side of the model which is X
    if who == 'x':
      for p in x.parameters:
        p.grad=None
      x.loss.backward()
      # optimizing...
      for p in x.parameters:
        p.data += (lr * p.grad)

    # with who passed as 'o' we only trained specific side of the model which is O
    if who == 'o':
      for p in o.parameters:
        p.grad=None
      o.loss.backward()
      # optimizing...
      for p in o.parameters:
        p.data += (lr * p.grad)
