using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Mail;
using System.Text;
using System.Threading.Tasks;
using BasicMultilayerPerceptron.DataSet;
using BasicMultilayerPerceptron.Integral;
using BasicMultilayerPerceptron.Perceptron;
using BasicMultilayerPerceptron.Perceptron.ActivationFunction;
using BasicMultilayerPerceptron.Perceptron.ErrorFunction;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {


            XOR();
            MNIST2();


       


        }


       public static void XOR()
        {
            BasicMultilayerPerceptron.DataSet.DataSet data = new BasicMultilayerPerceptron.DataSet.DataSet("xortrain.txt", ',', 1, false);
            Perceptron.Train(data, 293, 10000, 3, 1, 3, ActivationFunction.Sigmoid(),ErrorFunction.MSE());
 
            Console.ReadKey();
        }




        public static void MNIST2()
        {
            DataSet dataSet = new DataSet("mnist2.txt", ' ', 10, false);
            var p = new Perceptron(600, 3, ErrorFunction.CrossEntropy()).Layer(784, null).Layer(16, ActivationFunction.Sigmoid())
                .Layer(16, ActivationFunction.Sigmoid()).Layer(10, ActivationFunction.Sigmoid());
            p.Train2(dataSet, 200);
            double mse = p.CalculateMeanErrorOverDataSet(dataSet);
          
        }


        public static void MNIST()
        {
            BasicMultilayerPerceptron.DataSet.DataSet data = new BasicMultilayerPerceptron.DataSet.DataSet("mnist2.txt", ' ', 10, false);
            var p =Perceptron.Train(data, 60000, 10, 1, 3, 12, ActivationFunction.Sigmoid(),ErrorFunction.MSE());
            double mse = p.CalculateMeanErrorOverDataSet(data);
 
        }
    }
}
