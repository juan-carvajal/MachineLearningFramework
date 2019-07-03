using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Integral
{
    public  class DefiniteIntegral
    {

        public static double Calculate(Func<Double,Double>fx , double a , double b, double maxError)
        {
            int n = 2;
            double diff = Double.MaxValue;
            double lastVal=0;
            while(diff > maxError)
            {
                if (n == 2)
                {
                double val1 = CalculateWithN(n, fx, a, b);
                double val2 = CalculateWithN(n*2, fx, a, b);
                    diff =Math.Abs( val2 - val1);
                Console.WriteLine(n+" : "+diff);
                    lastVal = val2;
                    n *= 2;
                }
                else
                {
                    double val= CalculateWithN(n*2, fx, a, b);
                    diff = Math.Abs(val - lastVal);
                    Console.WriteLine(n + " : " + diff);
                    lastVal = val;
                    n *= 2;
                }
            }
            Console.WriteLine("Done in n = " + (n));
            return lastVal;
        }


        private static  double CalculateWithN(int n , Func<Double, Double> fx, double a, double b)
        {
            double result=0;
            double dX = (b - a) / (double)n;
            double currentX = a;
            while (currentX < b)
            {
                
                double nextX = currentX + dX;
                result += dX * ((fx.Invoke(currentX) + fx.Invoke(nextX)) / 2.0);
                currentX = nextX;
            }
            return result;
        }
    }
}
