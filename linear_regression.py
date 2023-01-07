# bach gradient descent

inputs = [1,2,3,4]
targets = [i*2 for i in inputs]

w = 0.1 # arbitraty initial value
learning_rate = 0.1

def predict(i): #equation of line f(x)=mx where m->w, i->x
    return w*i

# training the network
for iteration in range(25):
    pred = [predict(i) for i in inputs]
    errors = [t-p for p,t in zip(pred, targets)]
    cost = sum(errors) / len(targets) # indicate how closest are the approximations
    # print(f"iteration: {iteration + 1}, weight: {w:.2f}, cost: {cost:.2f}")
    w += cost * learning_rate # updating the weight in the network is called back propagation

# test the network
test_inputs = [5, 6]
test_targets = [10, 12]
pred = [predict(i) for i in test_inputs]

for i,t,p in zip(test_inputs, test_targets, pred):
    print(f"input: {i}, target: {t}, pred: {p:.4f}")