using System;
using System.Diagnostics;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Training
{
    public abstract class TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public INet<T> Net { get; protected set; }

        protected TrainerBase(INet<T> net)
        {
            this.Net = net;
        }

        public double BackwardTimeMs { get; protected set; }

        public double ForwardTimeMs { get; protected set; }

        public double UpdateWeightsTimeMs { get; protected set; }

        public virtual T Loss { get; protected set; }

        protected virtual void Backward(Volume<T> y)
        {
            var chrono = Stopwatch.StartNew();
            var batchSize = y.BatchSize;
            this.Loss = Ops<T>.Divide(this.Net.Backward(y), Ops<T>.Cast(batchSize));
            this.BackwardTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;
        }

        protected virtual Volume<T> Forward(Volume<T> x)
        {
            var chrono = Stopwatch.StartNew();
            var batchSize = x.BatchSize;
            var output = this.Net.Forward(x, true); // also set the flag that lets the net know we're just training
            this.ForwardTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;
            return output;
        }

        public virtual Volume<T> Train(Volume<T> x, Volume<T> y)
        {
            var batchSize = x.BatchSize;

            var output = Forward(x);

            Backward(y);

            var chrono = Stopwatch.StartNew();
            TrainImplem(x.BatchSize);
            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;

            return output;
        }

        protected abstract void TrainImplem(int batchSize);
    }
}