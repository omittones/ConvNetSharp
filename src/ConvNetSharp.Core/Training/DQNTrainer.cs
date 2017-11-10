using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    public class DQNTrainer : SgdTrainer<double>
    {
        private readonly Random rnd;
        private readonly int nmActions;
        private readonly Net<double> net;
        private readonly Experiences replayMemory;

        public int Samples { get; private set; }
        public double QValue { get; private set; }
        public int ReplayMemoryCount => replayMemory.Count;

        public double Gamma { get; set; }
        public double Epsilon { get; set; }
        public int ReplaySkipCount { get; set; }
        public int ReplaysPerIteration { get; set; }
        public double ClampErrorTo { get; set; }

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

            this.Gamma = 0.75; // future reward discount factor
            this.Epsilon = 0.1; // for epsilon-greedy policy
            this.LearningRate = 0.01; // value function learning rate
            this.ReplaySkipCount = 25; // number of time steps before we add another experience to replay memory
            this.ReplayMemorySize = 5000; // size of experience replay
            this.ReplayMemoryDiscardStrategy = ExperienceDiscardStrategy.First;
            this.ReplaysPerIteration = 10;
            this.ClampErrorTo = 1.0;

            this.nmActions = nmActions;

            this.net = net;

            this.Reset();
        }

        public void Reset()
        {
            this.Loss = 0;
            this.Samples = 0;

            // nets are hardcoded for now as key (str) -> Mat
            // not proud of this. better solution is to have a whole Net object
            // on top of Mats, but for now sticking with this
            this.replayMemory.Clear();
        }

        public Decision Act(double[] state)
        {
            var action = new Decision();
            action.State = (double[])state.Clone();

            // epsilon greedy policy
            if (rnd.NextDouble() < this.Epsilon)
            {
                action.Action = rnd.Next(0, this.nmActions);
            }
            else
            {
                // greedy wrt Q function
                var output = this.net.Forward(action.State, false);
                action.Action = output.IndexOfMax();
            }
            
            return action;
        }

        public double Learn(Decision decision, double[] nextState, double reward)
        {
            this.QValue = double.MinValue;

            // perform an update on Q function
            if (this.LearningRate > 0)
            {
                // learn from this tuple to get a sense of how "surprising" it is to the agent
                var xp = Experience.New(decision.State, decision.Action, reward, (double[])nextState.Clone());
                this.Loss = learnFromExperience(xp.state, xp.actionTaken, xp.reward, xp.nextState);

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
                    learnFromExperience(e.state, e.actionTaken, e.reward, e.nextState);
            }

            return this.Loss;
        }

        private double learnFromExperience(double[] s0, int a0, double r0, double[] s1)
        {
            this.BatchSize = 1;

            // want: Q(s,a) = r + gamma * max_a' Q(s',a')
            // compute the target Q value (current reward + gamma * next reward)
            if (Gamma != 0)
            {
                var a1vol = this.net.Forward(s1, false);
                var a1val = a1vol.IndexOfMax();
                var r1 = a1vol.Get(0, 0, a1val, 0);
                r0 += this.Gamma * r1;
            }
            
            if (this.QValue == double.MinValue)
                this.QValue = r0;

            // now predict
            var expected = this.Forward(s0).Clone();
            var a0val = expected.Get(0, 0, a0, 0);
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

            TrainImplem();

            return Math.Abs(error);
        }
    }
}
