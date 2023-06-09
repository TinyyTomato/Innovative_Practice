import impact_query_expert_finding.data.config
import os
import impact_query_expert_finding.data.sets
import impact_query_expert_finding.models
import impact_query_expert_finding.evaluation.batch
import impact_query_expert_finding.evaluation.visual
import pkg_resources


def run(parameters):
    # Load config file
    config_path = pkg_resources.resource_filename("impact_query_expert_finding", 'conf.yml')
    config = impact_query_expert_finding.data.config.load_from_yaml(config_path)

    #  Load parameters
    working_dir = parameters["output_dir"]

    #  Load dataset "aminer"
    dataset_path = parameters["input_dir"]
    dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
    dataset.load(dataset_path)


    eval_batch = impact_query_expert_finding.evaluation.batch.EvalBatch(dataset, dump_dir=working_dir,
                                                                        max_queries=parameters["max_queries"])

    models_dict = {
        "vote": impact_query_expert_finding.models.voting_model.VotingModel,
    }


    gathered_evaluations = dict()

    #  Get individual evaluations
    model_name = parameters["algorithm"]
    model = models_dict[model_name](config, **parameters)

    individual_evaluations = eval_batch.run_individual_evaluations(model, parameters=parameters)

    # Gather evaluations by all/topics
    gathered_evaluations[model_name] = eval_batch.merge_evaluations(individual_evaluations)

    # Plot prec/rec and ROC curves and other metrics for all evaluation
    impact_query_expert_finding.evaluation.visual.plot_evaluation(gathered_evaluations[model_name]["all"],
                                                                  prefix=model_name,
                                                                  path_visuals=working_dir, parameters=parameters)

    # Plot ROC curves for each topic
    # impact_query_expert_finding.evaluation.visual.plot_ROC_topics(gathered_evaluations[model_name]["topics"],
    # prefix=model_name,path_visuals=working_dir)

    # Plot pre rec curves for each topic
    # impact_query_expert_finding.evaluation.visual.plot_PreRec_topics(gathered_evaluations[model_name]["topics"], prefix=model_name,
    # path_visuals=working_dir)

# def run_multi(parameters):
#     # Load config file
#     config_path = pkg_resources.resource_filename("impact_query_expert_finding", 'conf.yml')
#     config = impact_query_expert_finding.data.config.load_from_yaml(config_path)
#
#     #  Load parameters
#     working_dir = parameters["output_dir"]
#     #  Load dataset "aminer"
#     dataset_path = parameters["input_dir"]
#     dataset = impact_query_expert_finding.data.sets.DataSet("aminer")
#     dataset.load(dataset_path)
#
#     eval_batch = impact_query_expert_finding.evaluation.batch.EvalBatch(dataset, dump_dir=working_dir,
#                                                                         max_queries=parameters["max_queries"])
#
#     models_dict = {
#         "vote": impact_query_expert_finding.models.voting_model.VotingModel,
#     }
#
#     print("Running evaluations for:", parameters["algorithm"])
#
#     gathered_evaluations = dict()
#
#     #  Get individual evaluations
#     model_name = parameters["algorithm"]
#     model = models_dict[model_name](config, **parameters)
#
#     individual_evaluations = eval_batch.run_individual_evaluations(model, parameters)
#
#     # Gather evaluations by all/topics
#     gathered_evaluations[model_name] = eval_batch.merge_evaluations(individual_evaluations)
#
#     # Plot prec/rec and ROC curves and other metrics for all evaluation
#     impact_query_expert_finding.evaluation.visual.plot_evaluation(gathered_evaluations[model_name]["all"],
#                                                                   prefix=model_name,
#                                                                   path_visuals=working_dir)
