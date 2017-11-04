using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ReinforcementLayer<T> : SoftmaxLayer<T>, IReinforcementLayer<T>, ILastLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        private int classCount;
        private T[] rewards;
        private int[] actionsTaken;

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            if (inputWidth != 1 || inputHeight != 1)
                throw new NotSupportedException();

            this.classCount = inputDepth;
        }

        //def discount_rewards(r):
        //""" take 1D float array of rewards and compute discounted reward """
        //discounted_r = np.zeros_like(r)
        //running_add = 0
        //for t in reversed(xrange(0, r.size)):
        //   running_add = running_add* gamma + r[t]
        //   discounted_r[t] = running_add
        //return discounted_r;

        private T[] DiscountedRewards(T[] rewards, double gamma)
        {
            T[] discounted = new T[rewards.Length];
            T running = Ops<T>.Zero;
            for (var i = rewards.Length - 1; i >= 0; i--)
            {
                running = Ops<T>.Multiply(running, Ops<T>.Cast(gamma));
                running = Ops<T>.Add(running, rewards[i]);
                discounted[i] = running;
            }
            return discounted;
        }

        public void SetLoss(int[] actionsTaken, T[] rewards)
        {
            this.rewards = rewards;
            this.actionsTaken = actionsTaken;
            if (this.InputActivationGradients.BatchSize != rewards.Length)
                throw new NotSupportedException("Output vs loss does not match!");
        }

        public override Volume<T> DoForward(Volume<T> input, bool isTraining = false)
        {
            return base.DoForward(input, isTraining);
        }

        public override Volume<T> Forward(bool isTraining)
        {
            return base.Forward(isTraining);
        }

        public override void Backward(Volume<T> y, out T expectedReward)
        {
            var batches = this.OutputActivation.BatchSize;

            expectedReward = Ops<T>.Zero;
            this.InputActivationGradients.Clear();
            for (var batch = 0; batch < batches; batch++)
            {
                //var action = actionsTaken[batch];
                for (var action = 0; action < this.classCount; action++)
                {
                    var policy = this.OutputActivation.Get(0, 0, action, batch);

                    var reward = rewards[batch];
                    expectedReward = Ops<T>.Add(expectedReward, Ops<T>.Multiply(reward, policy));

                    var derivative = Ops<T>.Subtract(Ops<T>.One, policy);
                    var advantageTimesDerivative = Ops<T>.Multiply(derivative, reward);
                    //advantageTimesDerivative = Ops<T>.Negate(advantageTimesDerivative);

                    this.InputActivationGradients.Set(0, 0, action, batch, advantageTimesDerivative);
                }
            }
        }
    }
}