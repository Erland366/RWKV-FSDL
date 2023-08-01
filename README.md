This is my **unofficial** repository for [RWKV Explained](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/) from [Full Stack Deep Learning](https://fullstackdeeplearning.com/).

The main purpose is that I want to learn to clean code in python file rather than using notebook for coding stuff. I also want to pair things from this code to the [paper](http://arxiv.org/abs/2305.13048) and try to explain my tought process of this code.

> 💡 P.S I am neither an AI researcher nor an AI engineer,. But I would like to be one (imagine me saying it like [this](https://www.youtube.com/shorts/qX-NxSfvhmQ))


The main thing that I learned are about `beartype` and `jaxtyping`. `jaxtyping` is used for documentation, while `beartype` checks if the typing from `jaxtyping` is correct in the code. I prefer not to use `beartype` because it may be troublesome for quick coding. Instead, I would use `jaxtyping` to clearly define variable types.

I placed all the types in `types.py` to keep them organized in one file, as I saw in this [repo](https://github.com/probml/dynamax) that I liked.

I also use simple config file using class `CFG` like in this [repo](https://github.com/kyegomez/RT-2/blob/main/train.py#L52). It's easy to implement for toy project like this. But I would probably use `hydra` for more complex project. I put `CFG` in `config.py` to make it cleaner.

To run the code, you can use the readily available code in `main.py`. It's also the final run code from the [article](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/).

## 📝 TODO
- [ ] Write explanation of each block based on the paper
- [ ] Replace the usage of `torch.exp` like the [article](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/) said
- [ ] Add clean `requirements.txt`
- [ ] (Optional) Try add `pytest` 