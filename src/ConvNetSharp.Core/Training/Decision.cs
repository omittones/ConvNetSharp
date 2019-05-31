namespace ConvNetSharp.Core.Training
{
    public struct Decision
    {
        public double[] State;
        public int Action;

        public Decision(int action, params double[] state)
        {
            Action = action;
            State = state;
        }
    }
}
