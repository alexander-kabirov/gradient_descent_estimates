from torch import tensor, rand, randn, randint, sum, empty, cat, max, arange, long, mm, optim, linalg, exp
from torch.nn import Module, parameter, functional, MSELoss


class Estimate(Module):
    """
    This simple model allows to test the estimation of latent variables given generated noisy observations.
    """
    def __init__(self, n_variables):  # , dropout=0.1
        super(Estimate, self).__init__()
        self.variables = parameter.Parameter(randn([n_variables, 1]), requires_grad=True)  # normally dist

    def forward(self, x) -> tensor:
        return sum(mm(x, exp(self.variables))).reshape([1])


def generate_training_data(sample_num: int, variable_num: int, sample_sum: int,
                           noise: bool = False, variable_min: float = 0.01, variance: float = 0.1) -> list:
    # generate random values for the latent variable with noise
    variables = rand(variable_num)
    indices = arange(variable_num)
    samples_ind = []
    sample_sum = sample_sum + randn([1]) if noise else sample_sum
    for i in range(sample_num):
        sample = empty(0)
        sample_ind = empty(0, dtype=long)
        remaining_value = sample_sum
        possible_vars = variables[variables <= remaining_value]
        possible_ind = indices[variables <= remaining_value]
        while len(possible_vars) > 0:
            ind = randint(len(possible_vars), [1])
            selected_var = possible_vars[ind]
            selected_ind = possible_ind[ind]
            if noise:
                selected_var = max(tensor([variable_min]), selected_var + randn([1])*variance)
            sample = cat((sample, selected_var))
            sample_ind = cat((sample_ind, selected_ind))
            remaining_value = sample_sum - sum(sample)
            possible_vars = variables[variables <= remaining_value]
            possible_ind = indices[variables <= remaining_value]
        samples_ind.append(functional.one_hot(sample_ind, num_classes=variable_num).float()) # with num_classes works ok
    return variables, samples_ind


if __name__ == '__main__':
    print('--> starting sample generation')
    variables, samples_ind = generate_training_data(100000, 10, 8, True)  # takes some time with large number of samples
    print('--> starting training')
    model = Estimate(10)
    print('--> model created')
    mse_loss = MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    for epoch in range(10):
        print(f'--> training epoch {epoch}')
        for sample in samples_ind:
            optimizer.zero_grad()
            output = model(sample)
            loss = mse_loss(output, tensor([8]).float())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        print(loss.item())

    print('Actual Variables:', variables, 'Estimated Variables:', exp(model.variables))
    print('Norm:', linalg.norm(variables-exp(model.variables)))  # check the norm to estimate the quality of the model