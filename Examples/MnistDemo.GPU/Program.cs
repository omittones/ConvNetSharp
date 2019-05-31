using System;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Single;
using ConvNetSharp.Core.Training.Single;
using ConvNetSharp.Volume.GPU.Single;

namespace MnistDemo.GPU
{
    internal class Program
    {
        private Net<float> net;
        private AdamTrainer _trainer;

        private static void Main()
        {
            var program = new Program();
            program.MnistDemo();
        }

        private void MnistDemo()
        {
            BuilderInstance.Volume = new VolumeBuilder();
            Ops<float>.SkipValidation = true;
            Ops<double>.SkipValidation = true;

            var datasets = new DataSets();
            if (!datasets.Load(0))
            {
                return;
            }

            // Create network
            this.net = new Net<float>();
            this.net.AddLayer(new InputLayer(28, 28, 1));
            this.net.AddLayer(new ConvLayer(4, 4, 32) { Stride = 1, Pad = 1 });
            this.net.AddLayer(new PoolLayer(2, 2) { Stride = 1 });
            this.net.AddLayer(new LeakyReluLayer(0.3f));
            this.net.AddLayer(new ConvLayer(4, 4, 8) { Stride = 1, Pad = 1 });
            this.net.AddLayer(new PoolLayer(2, 2) { Stride = 1 });
            this.net.AddLayer(new LeakyReluLayer(0.3f));
            this.net.AddLayer(new FullyConnLayer(300));
            this.net.AddLayer(new LeakyReluLayer(0.3f));
            this.net.AddLayer(new FullyConnLayer(150));
            this.net.AddLayer(new LeakyReluLayer(0.3f));
            this.net.AddLayer(new FullyConnLayer(10));
            this.net.AddLayer(new SoftmaxLayer(10));

            this._trainer = new AdamTrainer(this.net);

            var random = new Random();
            var @params = this.net.GetParametersAndGradients().ToArray();
            foreach (var param in @params)
            {
                for (var i = 0; i < param.Volume.Shape.TotalLength; i++)
                    param.Volume.Set(i, (float)(random.NextDouble() * 0.0001 - 0.00005));
            }

            var totalSamples = datasets.Train.Count;

            //if (File.Exists("loss.csv"))
            //{
            //    File.Delete("loss.csv");
            //}

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            Batch sample = null;
            int currentSamples = 0;
            int epoch = 0;
            do
            {
                sample = datasets.Train.NextBatch(500, sample);
                var loss = Train(sample);

                currentSamples += sample.Item3.Length;

                int progress = (int)(50.0 * currentSamples / totalSamples);
                var bar = new string('-', progress) + ">" + new string('.', 50 - progress);
                bar += $" {currentSamples}/{totalSamples}  loss: {loss}     \r";
                Console.Write(bar);

                if (sample.Final)
                {
                    var trainAcc = Accuracy(datasets.Train);
                    var testAcc = Accuracy(datasets.Test);

                    Console.WriteLine();
                    Console.WriteLine($"Epoch {epoch} - loss: {loss}, train accuracy: {trainAcc:0.00}%, test accuracy: {testAcc:0.00}%");
                    Console.WriteLine($"     fw: {forwardTime}");
                    Console.WriteLine($"     bw: {backwardTime}");
                    Console.WriteLine($"     up: {updateTime}");

                    //File.AppendAllLines("loss.csv",
                    //new[]
                    //{
                    //    $"{this._stepCount}, {this._trainer.Loss}, {Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2)}, {Math.Round(this._testAccWindow.Items.Average() * 100.0, 2)}"
                    //});

                    forwardTime = 0;
                    backwardTime = 0;
                    updateTime = 0;
                    currentSamples = 0;
                    epoch++;
                }

            } while (!Console.KeyAvailable);
        }

        private double Accuracy(DataSet set)
        {
            int correct = 0;
            int total = 0;
            Batch batch = null;
            while (true)
            {
                batch = set.NextBatch(500, batch);

                total += batch.Item3.Length;

                this.net.Forward(batch.Item1);
                var prediction = this.net.GetPrediction();
                for (var i = 0; i < batch.Item3.Length; i++)
                    correct += batch.Item3[i] == prediction[i] ? 1 : 0;

                if (batch.Final)
                    return 100.0 * correct / total;
            }
        }

        private double forwardTime = 0.0;
        private double backwardTime = 0.0;
        private double updateTime = 0.0;

        private float Train(Batch batch)
        {
            this._trainer.Train(batch.Item1, batch.Item2);
            forwardTime += this._trainer.ForwardTimeMs;
            backwardTime += this._trainer.BackwardTimeMs;
            updateTime += this._trainer.UpdateWeightsTimeMs;
            return this._trainer.Loss;
        }
    }
}