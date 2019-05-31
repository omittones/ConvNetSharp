using ConvNetSharp.Core.Serialization;
using ConvNetSharp.Volume;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    public class Experience
    {
        public double Reward;
        public double[] State;
        public double[] NextState;
        public int ActionTaken;

        internal static Experience New(double[] s0, int a0, double r0, double[] s1)
        {
            return new Experience
            {
                State = (double[])s0.Clone(),
                NextState = (double[])s1.Clone(),
                ActionTaken = a0,
                Reward = r0
            };
        }

        public override string ToString()
        {
            return $"({State.ToHumanString("{0:0.000}")},{ActionTaken}) -> ({NextState.ToHumanString("{0:0.000}")}) with reward({Reward:0.0000})";
        }
    }
}
