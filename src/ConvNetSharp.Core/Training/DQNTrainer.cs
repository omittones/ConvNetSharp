using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;
using System.Diagnostics;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    public class DQNTrainer : SgdTrainer<double>
    {
        private INet<double> trainee;
        private INet<double> freezed;

        private readonly Random rnd;
        private readonly int nmActions;
        private readonly Experiences replayMemory;
        public IEnumerable<Experience> ReplayMemory => replayMemory;
        public int ReplayMemoryCount => replayMemory.Count;

        public int Samples { get; private set; }
        public double QValue { get; private set; }
        public double Gamma { get; set; }
        public double Epsilon { get; set; }
        public int ReplaySkipCount { get; set; }
        public int ReplaysPerIteration { get; set; }
        public double ClampErrorTo { get; set; }
        public double? MaxQValue { get; set; }
        public double? MinQValue { get; set; }
        public int FreezeInterval { get; set; }

        public static double TheoreticalMaxQValue(double gamma, double maxReward)
        {
            if (gamma >= 1.0)
                return double.PositiveInfinity;
            var max = maxReward / (1.0 - gamma);
            if (max < 0)
                max = 0;
            return max;
        }

        public static double TheoreticalMinQValue(double gamma, double minReward)
        {
            if (gamma >= 1.0)
                return double.NegativeInfinity;
            var min = minReward / (1.0 - gamma);
            if (min > 0)
                min = 0;
            return min;
        }

        public int ReplayMemorySize
        {
            get => this.replayMemory.Size;
            set => this.replayMemory.Size = value;
        }

        public ExperienceDiscardStrategy ReplayMemoryDiscardStrategy
        {
            get => replayMemory.DiscardStrategy;
            set => replayMemory.DiscardStrategy = value;
        }

        public DQNTrainer(
            Net<double> net,
            int nmActions) : base(net)
        {
            this.rnd = new Random(DateTime.Now.Millisecond);
            this.replayMemory = new Experiences();

            this.LearningRate = 0.01; // value function learning rate
            this.Epsilon = 0.1; // for epsilon-greedy policy
            this.Gamma = 0.75; // future reward discount factor
            this.ReplaySkipCount = 25; // number of time steps before we add another experience to replay memory
            this.ReplayMemorySize = 5000; // size of experience replay
            this.ReplayMemoryDiscardStrategy = ExperienceDiscardStrategy.First;
            this.ReplaysPerIteration = 100;
            this.ClampErrorTo = double.MaxValue;
            this.FreezeInterval = 1;

            this.nmActions = nmActions;

            this.freezed = net;
            this.trainee = net;

            this.Reset();
        }

        public void Reset()
        {
            this.Loss = 0;
            this.Samples = 0;
            this.replayMemory.Clear();
        }

        public Decision Act(double[] state)
        {
            EnsureInitialized();

            state = (double[])state.Clone();

            var action = 0;

            // epsilon greedy policy
            if (rnd.NextDouble() < this.Epsilon)
            {
                action = rnd.Next(0, this.nmActions);
            }
            else
            {
                // greedy wrt Q function
                var output = this.freezed.Forward(state, false);
                action = output.IndexOfMax();
            }

            return new Decision(action, state);
        }

        private void EnsureInitialized()
        {
            if (FreezeInterval > 1 && object.ReferenceEquals(this.freezed, this.trainee))
            {
                this.Net = this.freezed.Clone();
                this.trainee = this.Net;
            }
            else if (FreezeInterval <= 1 && object.ReferenceEquals(this.freezed, this.trainee))
            {
                this.Net = this.freezed;
                this.trainee = this.freezed;
            }
        }

        public double Learn(Decision decision, double[] nextState, double reward)
        {
            this.QValue = double.MinValue;

            // perform an update on Q function
            if (this.LearningRate > 0)
            {
                // learn from this tuple to get a sense of how "surprising" it is to the agent
                var xp = Experience.New(decision.State, decision.Action, reward, nextState);
                this.Loss = learnFromExperience(xp.State, xp.ActionTaken, xp.Reward, xp.NextState);

                // decide if we should keep this experience in the replay
                if (this.ReplaySkipCount < 1 ||
                    this.Samples % this.ReplaySkipCount == 0)
                {
                    // roll over when we run out
                    this.replayMemory.Add(xp);
                }
                this.Samples += 1;

                var trainingSet = new List<Experience>();
                if (this.ReplaysPerIteration > this.replayMemory.Count)
                {
                    trainingSet.AddRange(this.replayMemory);
                }
                else
                {
                    // sample some additional experience from replay memory and learn from it
                    for (var k = 0; k < this.ReplaysPerIteration; k++)
                    {
                        var ri = this.rnd.Next(0, this.replayMemory.Count); // todo: priority sweeps?
                        trainingSet.Add(this.replayMemory[ri]);
                    }
                }

                foreach (var e in trainingSet)
                    learnFromExperience(e.State, e.ActionTaken, e.Reward, e.NextState);
            }

            if (!object.ReferenceEquals(this.trainee, this.freezed) &&
                this.FreezeInterval > 1 &&
                Samples % this.FreezeInterval == 0)
            {
                this.trainee.CopyParameters(this.freezed);
            }

            return this.Loss;
        }

        private double adjustQValuePerSettings(double value)
        {
            if (this.MaxQValue.HasValue && value > this.MaxQValue)
                value = this.MaxQValue.Value;

            if (this.MinQValue.HasValue && value < this.MinQValue)
                value = this.MinQValue.Value;

            return value;
        }

        private double learnFromExperience(double[] s0, int a0, double r0, double[] s1)
        {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')
            // compute the target Q value (current reward + gamma * next reward)
            if (Gamma != 0)
            {
                var a1vol = this.freezed.Forward(s1, false);
                var a1val = a1vol.IndexOfMax();
                var r1 = a1vol.Get(0, 0, a1val, 0);
                r0 += this.Gamma * r1;
                r0 = adjustQValuePerSettings(r0);
            }

            if (this.QValue == double.MinValue)
                this.QValue = r0;

            // now predict
            var expected = this.Forward(s0).Clone();
            var a0val = expected.Get(0, 0, a0, 0);

            a0val = adjustQValuePerSettings(a0val);

            var error = a0val - r0;

            // huber loss to robustify
            var clampedError = error;
            if (clampedError > this.ClampErrorTo)
                clampedError = this.ClampErrorTo;
            if (clampedError < -this.ClampErrorTo)
                clampedError = -this.ClampErrorTo;

            //pred.dw[a0] = tderror;
            expected.Set(0, 0, a0, 0, a0val - clampedError);

            //propagate errors
            this.Backward(expected); // compute gradients on net params

            TrainImplem(1);

            return Math.Abs(error);
        }
    }
}
