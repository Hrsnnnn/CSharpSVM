using System;
using LibSVMsharp;
using LibSVMsharp.Helpers;
using LibSVMsharp.Extensions;

namespace ConsoleApp1 {
    class Program {
        static void Main(string[] args) {
            Console.WriteLine("This is trainer training");
            SVMProblem problem = SVMProblemHelper.Load(@"D:/WorkSpace/CSharpSVM/ConsoleApp1/train_data.txt");
            SVMProblem testProblem = SVMProblemHelper.Load(@"D:/WorkSpace/CSharpSVM/ConsoleApp1/test_data.txt");
            // SVMProblem p = SVMProblemHelper.Load(@"D:/WorkSpace/CSharpSVM/ConsoleApp1/a.txt");

            // Console.WriteLine(problem.X[0][0].Value);
            // Console.WriteLine(p.Y[0]);

            SVMParameter parameter = new SVMParameter();
            parameter.Type = SVMType.C_SVC;
            parameter.Kernel = SVMKernelType.RBF;
            parameter.C = 1;
            parameter.Gamma = 1;

            // Console.WriteLine(problem.Length);
            // SVMModel model = SVM.Train(testProblem, parameter);
            SVMModel model = problem.Train(parameter);
            // for(int i=0;i<model.Labels.Length;i++)
            //     Console.WriteLine(model.SVCoefs.ToString());
            // Console.WriteLine(model.ToString());
            double[] target = testProblem.Predict(model);
            // for(int i=0;i<target.Length;i++)
            //     Console.WriteLine(target[i].ToString() + " " + i.ToString());
            double accuracy = testProblem.EvaluateClassificationProblem(target);

            /*double[] target = new double[testProblem.Length];
            for (int i = 0; i < testProblem.Length; i++) {
                target[i] = SVM.Predict(model, testProblem.X[i]);
                Console.WriteLine(target[i].ToString() + " " + problem.Y[i] + " " + i.ToString() + " " + problem.X[i][0].Value);
            }*/
            // double accuracy = SVMHelper.EvaluateClassificationProblem(testProblem, target);
            Console.WriteLine(accuracy);
        }
    }
}
