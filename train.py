from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from tqdm import tqdm
import visual
import utils


def train(model, train_dataset, test_dataset=None, collate_fn=None,
          model_dir='models', lr=1e-3, lr_decay=.1,
          lr_decay_epochs=None, weight_decay=1e-04, grad_clip_norm=10.,
          batch_size=32, test_size=256, epochs=5,
          eval_log_interval=30,
          gradient_log_interval=50,
          loss_log_interval=30,
          checkpoint_interval=500,
          resume_best=False,
          resume_latest=False,
          cuda=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = MultiStepLR(optimizer, lr_decay_epochs, gamma=lr_decay)

    model.train()
    epoch_start = 1
    best_precision = 0

    if resume_best or resume_latest:
        epoch_start, best_precision = utils.load_checkpoint(
            model, model_dir, best=resume_best
        )

    for epoch in range(epoch_start, epochs+1):
        scheduler.step(epoch-1)
        data_loader = utils.get_data_loader(
            train_dataset, batch_size,
            cuda=cuda, collate_fn=collate_fn
        )
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, q, a) in data_stream:
            # where are we?
            data_size = len(x)
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(dataset_size // batch_size) +
                batch_index + 1
            )

            x = Variable(x).cuda() if cuda else Variable(x)
            q = Variable(q).cuda() if cuda else Variable(q)
            a = Variable(a).cuda() if cuda else Variable(a)

            optimizer.zero_grad()
            scores = model(x, q)
            loss = criterion(scores, a)
            loss.backward()

            _, predicted = scores.max(1)
            precision = (predicted == a).sum().data[0] / data_size

            if grad_clip_norm:
                nn.utils.clip_grad_norm(model.parameters(), grad_clip_norm)
            optimizer.step()

            # update & display statistics.
            data_stream.set_description((
                'epoch: {epoch}/{epochs} | '
                'total iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'prec: {prec:.4} | '
                'loss: {loss:.4} '
            ).format(
                epoch=epoch,
                epochs=epochs,
                iteration=iteration,
                trained=(batch_index+1)*batch_size,
                total=dataset_size,
                progress=(100.*(batch_index+1)/dataset_batches),
                prec=precision,
                loss=loss.data[0],
            ))

            # Send gradient norms to the visdom server.
            if iteration % gradient_log_interval == 0:
                names, gradients = zip(*[
                    (n, p.grad.norm().data) for
                    n, p in model.named_parameters()
                ])
                visual.visualize_scalars(
                    gradients, names, 'gradient l2 norms',
                    iteration, env=model.name
                )

            # Send test precision to the visdom server.
            if iteration % eval_log_interval == 0:
                visual.visualize_scalar(utils.validate(
                    model, test_dataset,
                    test_size=test_size, cuda=cuda,
                    collate_fn=collate_fn, verbose=False
                ), 'precision', iteration, env=model.name)

            # Send losses to the visdom server.
            if iteration % loss_log_interval == 0:
                visual.visualize_scalar(
                    loss.data / data_size,
                    'loss', iteration, env=model.name
                )

            if iteration % checkpoint_interval == 0:
                # notify that we've reached to a new checkpoint.
                print()
                print()
                print('#############')
                print('# checkpoint!')
                print('#############')
                print()

                # test the model.
                model_precision = utils.validate(
                    model, test_dataset or train_dataset,
                    test_size=test_size, cuda=cuda,
                    collate_fn=collate_fn, verbose=True
                )

                # update best precision if needed.
                is_best = model_precision > best_precision
                best_precision = max(model_precision, best_precision)

                # save the checkpoint.
                utils.save_checkpoint(
                    model, model_dir, epoch,
                    model_precision, best=is_best
                )
                print()
