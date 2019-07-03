using System;
using System.Collections.Generic;
using BasicMultilayerPerceptron.DataSet;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron
{
    public class Perceptron
    {

        private List<Layer> Layers;
        private int Batching;
        private double LearningRate;
        
        private ErrorFunction.ErrorFunction ErrorFunction;

        public Perceptron(int batching , double learningRate ,  ErrorFunction.ErrorFunction errorFunction )
        {
            if(batching < 1)
            {
                throw new ArgumentException("Parameter batching must be a positive integer.");
            }

            if(learningRate <= 0)
            {
                throw new ArgumentException("Parameter learningRate must be a positive real number.");
            }

            
            this.Batching = batching;
            this.LearningRate = learningRate;
            this.Layers = new List<Layer>();
            this.ErrorFunction = errorFunction;
        }


        public Perceptron Layer (int size , ActivationFunction.ActivationFunction activationFunction)
        {
            if(Layers.Count == 0)
            {
                Layer layer = new Layer(activationFunction, size);
                this.Layers.Add(layer);
            }
            else
            {
                Layer lastLayer = Layers[Layers.Count - 1];
                Layer layer = new Layer(activationFunction, size);
                this.Layers.Add(layer);
                lastLayer.Next = layer;
                layer.Previous = lastLayer;
                layer.BuildWeightMatrix();


            }
            return this;
        }

        public double[] FeedForward(double[] data)
        {
            this.Layers[0].Activate(data);
            return this.Layers[this.Layers.Count - 1].Activations;
        }


        public double CalculateExampleLost(DataRow example)
        {
            double[] estimators = this.FeedForward(example.Features);
            return  example.Labels.Zip(estimators, (l, e) => ErrorFunction.GetValue(l, e)).Sum();
        }


        public double MeanLossOverDataSet(DataSet.DataSet dataSet)
        {
            return dataSet.DataRows.Select(i => CalculateExampleLost(i)).Average();
        }


        public double CalculateMeanErrorOverDataSet(DataSet.DataSet dataSet)
        {
            double acum = 0;
            int cont = 0;
            foreach(var row in dataSet.DataRows)
            {
                if(cont%(dataSet.DataRows.Count / 10) == 0)
                {
                //Console.WriteLine("     " + cont);

                }
                cont++;
                var err= this.FeedForward(row.GetFeatures()).Zip(row.GetLabels(), (e, l) => ErrorFunction.GetValue(l,e)).Sum();

                acum += err;
            }
            acum /= (double)dataSet.DataRows.Count;
            return acum;
        }


        private void TakeGradientDescentStep(int miniBatchSize)
        {
            //Console.WriteLine("Started step...");
            foreach (Layer l in this.Layers)
            {
                if (l.Previous != null)
                {
                    for (int i = 0; i < l.BiasVector.Length; i++)
                    {
                        l.BiasVector[i] -= this.LearningRate * (l.BiasVectorChangeRecord[i] / (double)miniBatchSize);
                    }

                    for (int i = 0; i < l.WeightMatrix.GetLength(0); i++)
                    {
                        for (int j = 0; j < l.WeightMatrix.GetLength(1); j++)
                        {
                            l.WeightMatrix[i, j] -= this.LearningRate * (l.WeightMatrixChangeRecord[i, j] / (double)miniBatchSize);
                        }
                    }

                }

            }
            FlushChangeRecords();
            //Console.WriteLine("Finished step.");
        }


        private void FlushChangeRecords()
        {
            foreach(Layer l in this.Layers)
            {
                if (l.Previous != null)
                {

                    l.BiasVectorChangeRecord = new double[l.BiasVectorChangeRecord.Length];
                    l.WeightMatrixChangeRecord = new double[l.WeightMatrixChangeRecord.GetLength(0), l.WeightMatrixChangeRecord.GetLength(1)];
                }
            }
        }


      


        private void Train(DataSet.DataSet dataSet ,int epochs)
        {
            Console.WriteLine("MSE:" + CalculateMeanErrorOverDataSet(dataSet));
            for (int i = 0; i < epochs; i++)
            {
                
                dataSet.Shuffle();
                List<List<DataRow>> batch = dataSet.Batch(this.Batching);
                int step = 0;
                foreach (List<DataRow> row in batch)
                {
                    foreach(DataRow example in row)
                    {
                        double[] result = this.FeedForward(example.GetFeatures());
                        double[] labels = example.GetLabels();
                        if (result.Length != labels.Length)
                        {
                            throw new Exception("Inconsistent array size, Incorrect implementation.");
                        }
                        else
                        {
                            double error = labels.Zip(result, (x, y) =>  Math.Pow(x - y,2) ).Sum();
                            for (int l =this.Layers.Count-1; l >0; l--)
                            {
                                if (l == this.Layers.Count - 1)
                                {
                                    for (int j = 0; j < this.Layers[l].CostDerivatives.Length; j++)
                                    {
                                        this.Layers[l].CostDerivatives[j] = 2.0 * (this.Layers[l].Activations[j] - labels[j]);
                                    }

                                }
                                else
                                {
                                    for (int j = 0; j < this.Layers[l].CostDerivatives.Length; j++)
                                    {
                                        //this.Layers[l].CostDerivatives[j] = 2.0 * (this.Layers[l].Activations[j] - labels[j]);
                                        double acum = 0;
                                        for (int j2 = 0; j2 < Layers[l+1].Size; j2++)
                                        {
                                            acum += Layers[l + 1].WeightMatrix[j2, j] * Layers[l + 1].ActivationFunction.GetDerivativeValue(Layers[l + 1].WeightedSum[j2]) * Layers[l + 1].CostDerivatives[j2];
                                        }
                                        this.Layers[l].CostDerivatives[j] = acum;
                                    }
                                }

                                for (int j = 0; j < this.Layers[l].Activations.Length; j++)
                                {
                                    this.Layers[l].BiasVectorChangeRecord[j] += Layers[l].ActivationFunction.GetDerivativeValue(Layers[l].WeightedSum[j]) * Layers[l].CostDerivatives[j];
                                    for (int k = 0; k < Layers[l].WeightMatrix.GetLength(1); k++)
                                    {
                                        this.Layers[l].WeightMatrixChangeRecord[j, k] += Layers[l - 1].Activations[k]
                                            * Layers[l].ActivationFunction.GetDerivativeValue(Layers[l].WeightedSum[j])
                                            * Layers[l].CostDerivatives[j];
                                    }
                                }
                            }
                        }

                    }
                   // Console.WriteLine("Step "+step);
                    step++;
                    TakeGradientDescentStep(row.Count);
                    
                    //
                }
                Console.WriteLine(i + ":" + CalculateMeanErrorOverDataSet(dataSet));
            }
        }



        public void Train2(DataSet.DataSet dataSet, int epochs)
        {
            Console.WriteLine("Initial Loss:"+ CalculateMeanErrorOverDataSet(dataSet));
            for (int i = 0; i < epochs; i++)
            {
                dataSet.Shuffle();
                List<DataRow> batch = dataSet.NextBatch(this.Batching);

                int count = 0;
                    foreach (DataRow example in batch)
                    {

                    count++;

                    double[] result = this.FeedForward(example.GetFeatures());
                        double[] labels = example.GetLabels();
                        if (result.Length != labels.Length)
                        {
                            throw new Exception("Inconsistent array size, Incorrect implementation.");
                        }
                        else
                        {
                            double error = CalculateExampleLost(example);

                        
                            for (int l = this.Layers.Count - 1; l > 0; l--)
                            {
                                if (l == this.Layers.Count - 1)
                                {
                                    for (int j = 0; j < this.Layers[l].CostDerivatives.Length; j++)
                                    {
                                    this.Layers[l].CostDerivatives[j] = ErrorFunction.GetDerivativeValue(labels[j], this.Layers[l].Activations[j]);

                                    }

                                }
                                else
                                {
                                    for (int j = 0; j < this.Layers[l].CostDerivatives.Length; j++)
                                    {

                                        double acum = 0;
                                        for (int j2 = 0; j2 < Layers[l + 1].Size; j2++)
                                        {
                                            acum += Layers[l + 1].WeightMatrix[j2, j] * this.Layers[l+1].ActivationFunction.GetDerivativeValue(Layers[l + 1].WeightedSum[j2]) * Layers[l + 1].CostDerivatives[j2];
                                        }
                                        this.Layers[l].CostDerivatives[j] = acum;
                                    }
                                }

                                for (int j = 0; j < this.Layers[l].Activations.Length; j++)
                                {
                                    this.Layers[l].BiasVectorChangeRecord[j] += this.Layers[l].ActivationFunction.GetDerivativeValue(Layers[l].WeightedSum[j]) * Layers[l].CostDerivatives[j];
                                    for (int k = 0; k < Layers[l].WeightMatrix.GetLength(1); k++)
                                    {
                                        this.Layers[l].WeightMatrixChangeRecord[j, k] += Layers[l - 1].Activations[k]
                                            * this.Layers[l].ActivationFunction.GetDerivativeValue(Layers[l].WeightedSum[j])
                                            * Layers[l].CostDerivatives[j];
                                    }
                                }
                            }
                        }

                    
                    
                    }
                    TakeGradientDescentStep(batch.Count);

                if ((i + 1) % (epochs / 10) == 0)
                {
                    Console.WriteLine("Epoch " + (i + 1) + ", Avg.Loss:" + CalculateMeanErrorOverDataSet(dataSet));
                }
            }

        }

        public void Train3(DataSet.DataSet dataSet, int epochs)
        {
            for (int i = 0; i < epochs; i++)
            {
                var miniBatch = dataSet.NextBatch(this.Batching);
                foreach(var example in miniBatch)
                {
                    double error = CalculateExampleLost(example);
                    double[] labels = example.Labels;

                    Layers.Last().BackPropagate(error, labels, ErrorFunction);
                }
                TakeGradientDescentStep(miniBatch.Count);
                if ((i + 1) % (epochs / 10) == 0)
                {
                    Console.WriteLine("Epoch " + (i + 1) + ", Avg.Loss:" + CalculateMeanErrorOverDataSet(dataSet));
                }
            }
        }


        public static Perceptron Train(DataSet.DataSet dataSet , int batching,int epochs , double learningRate, int hiddenLayers, int hiddenLayersSize,ActivationFunction.ActivationFunction activationFunction , ErrorFunction.ErrorFunction errorFunction)
        {
            Perceptron p = new Perceptron(batching, learningRate,errorFunction);
            p.Layer(dataSet.FeatureSize,activationFunction);
            for (int i = 0; i < hiddenLayers; i++)
            {
                p.Layer( hiddenLayersSize,activationFunction);
            }
            p.Layer(dataSet.LabelSize, activationFunction);

            p.Train2(dataSet, epochs);

            return p;
        }


    }
}
