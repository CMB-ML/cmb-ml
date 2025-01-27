# Notes to Self

I'm imagining a series of notebooks continuing the tutorial.
- CMB-ML patterns
- A pipeline for ML systems (or description of setting one up)
- A pipeline for classic algorithms
- The download notebook with guts exposed

This CMB-ML patterns notebook should cover:
- The pipeline as a whole
- An Executor
- An Asset
- An AssetHandler
- The Namer
- The pipeline yaml

Next notebook
- Putting it together: Dequangling

# Introduction / Challenges

I really struggled to manage that data early on. In many repositories, a single set of operations is applied to a single file. Afterwards, the script stops. That works well for those systems, but when I tried to emulate those patterns across multiple stages and across multiple simulations... it quickly became a tangled mess.

(Describe issues once here, then reference them later.)

For instance, I create a file in one place and need to read it in another. When I separate my code into chunks, I often found myself torn between creating an object at a high level and passing it down, or recreating the object in multiple places in the software. The former makes code ungainly, and the latter makes it unreliable.

Alternative: describe in bullet points

I also struggled with dataset splits. I have a vision of being able to test on subcategories of data, which span define-able subspaces of the full distribution. I want to keep those segmented. 

# Pipeline

Description of the pipeline. Figure Here. Don't worry about the details. Let's imagine a simpler pipeline with just a couple stages.

Producing simulations, cleaning the signal, and performing analysis on hundreds of simulations has proven challenging. Each of those three phases can be further broken down into many stages. Some stages consistently follow similar steps, though others need to do things differently. We're still figuring out overall templates for these (which means there's an unfortunate amount of boilerplate).

We implemented this pipeline system in order to:
- keep track of variables
- limit repetition of file paths
- provide consistent interface to stages

I define a few different classes to manage these stages. I'll introduce them here, dive a bit deeper on each, then provide a small example to tie it together. I'm still trying to figure this stuff out -- I've spent a while on it, and I think I've arrived at a good solution, but I'm still on somewhat shaky ground and want to be open to making a firmer foundation to this.

To make it simpler, each stage is seen as a simple function: **data** and **parameters** go in, and different **data** comes out.

# Patterns Overview

A stage is managed by an Executor, each defined in a class with a repeated pattern. At initialization, the **parameters** are made local variables within the Executor. References for the **data**, both input and output, are stored. We track those references as Assets. Data may be stored in different formats - sometimes maps in ".fits" files, sometimes Numpy arrays, and sometimes plain text. Instead of handling each file type separately, we instead have a single Asset class, and multiple AssetHandler's, for each file type.

This may seem over-wrought, but it enables calling the Executor a single time, and allowing it to handle all the different simulations which are to be processed. Thus, each Asset refers not to a single file, but to a class of files.  Consider a hypothetical stage that produces a CMB map from a power spectrum. We simply load the power spectrum, feed it to a function, and get the map out. There is an Asset for the CMB map and another Asset for the power spectrum. This Executor needs to process many power spectra and produce many maps. Instead of recreating an Asset for each input file name, I take a different approach.

We instead use a Namer object, a member of the Executor for a particular stage. The Namer has a state and as the Executor moves between simulations, the state of the Namer is changed accordingly. Each Asset references the Namer when it is called to read or write a file.

That does it for a stage, but what about the structure tying all the stages together?

In the top-level scripts, a PipelineContext is created. Each Executor is added, starting with an Executor (or two) to check over the configuration file for conflicts. In any of these scripts, the order of the stages can be quickly reviewed. The pipeline context then has a "preliminary run" (prerun_pipeline()), which simply instantiates each Executor in order to ensure that the configuration files and Executors are in agreement (please don't load maps during an __init__() method).

A couple other helper classes are also provided. One handles logging. <!-- TODO: OTHERS???? --> Splits.

We'll discuss these in more detail below, and provide examples.

# Configuration yamls

The Pipeline yamls contain information for each stage.

The Local System yaml contains the path to root folders on your file system. These are used when composing paths to files.

The File System yaml contains defaults which can be used; the current revision has them removed. Unknown if they should be added back.

The remaining yamls contain more generic parameters. Scenario describes the experiment setup (though it may be vestigial at this point). Splits describes the data splits - Training, Validation, and Test.

## Pipeline YAMLs

The pipeline YAMLs follow a template. Each describes the set of stages for a particular phase. For each stage, we have a name, output assets, input assets, the name of the directory which will contain its output, the splits which are processed, and some ancilliary settings.

Output assets must define a handler and a path_template. Input assets instead refer to the stage of the pipeline that originally created these assets. This keeps the assets consistent. There are a few things which come from external sources like the WMAP chains and Planck's maps for variance and masks; these are defined in the sim pipeline yaml, under `raw`. 

In order for the later phases to refer to assets produced in earlier phases, there are "assembly_" pipeline yamls. These simply import the relevant phases.

# Patterns in Depth

For each of the classes, I'll (1) describe it in a bit more detail, (2) show how they're set up in both Python and the configuration files, and (3) show how they're used.

## Assets

### Assets Overview

At its most generic, data is read from the files, used in operations, then some new data is saved. 
That's not the end of the use of that data, however. 
The same maps written in one stage of the pipeline will need to be read in others.
I want to provide a generic way to operate on the data with standard interfaces.
This is done with an Asset.

In notebook A, we introduced Hydra which manages parameters for CMB-ML. In notebook C, we set up your local system and the file paths used. These file paths can be seen as *parameters*, but they're really just pointing at what we really want: the **data** in the files.

An Asset has the single responsibility to enable reading or writing data. To do so, it has two methods, `read()` and `write()`. Each Asset is set up automatically when an Executor is created. To avoid confusion, each Asset is either an input or an output - if it can `read()`, it cannot `write()` (and vice-versa). This is critical, as otherwise an executor can run multiple times and put the dataset into an unknown state.

An Asset will also have a `path` property which returns its location on disk (or prospective location).

There is only one class for Assets. To handle different file types, we have many different AssetHandlers. When read() or write() is called, the AssetHandler is used.

### Asset Set-Up

For a stage, the pipeline yaml defines `assets_out` and `assets_in`. When the Executor is created, it will pull from the config and generate two lists: `self.assets_out` and `self.assets_in`. 

[Example yaml here]

Output assets are fully defined in the configuration. They have a handler and a path_template. The Handler is the name of an AssetHandler object. The path_template is a string with tags that will be filled in by the Namer. A common source of errors is using the wrong tag label. The ones I've used so far are:
- root
- src_root (as an alternative to root, when pulling from external data products)
- dataset
- working
- stage
- split
- sim
- freq
- epoch

Input assets are not redefined. We've defined them already for a previous stage. Instead, we refer back to that stage. For instance, in make_sims, we have noise_maps: {stage: make_noise}. When the asset is set up, the source stage is used.

On occasion, I want to simplify a name. For instance, in make_mask, I have an input asset `mask: {stage: raw, orig_name: mask_src_map}`. The `orig_name` means that in the raw stage I define a mask_src_map, but when I create the asset, I will put it into my assets_in with the key 'mask'.

Internally, when the assets are set up, they are given access to the Executor's Namer object. When the path is requested, the Namer will use its current context to fill in the tag labels. This will become more clear through examples and the description of the Executor (hoepfully).

### Using Assets

When putting together the Executor, suppose I have `my_input_asset = self.assets_in['my_input']`. Later, I can simply have `my_input = my_input_asset.read()`. Some of the assets need more parameters, such as the HealpyMap can have, for instance, multiple fields. Refer to the particular AssetHandler subclass to find these.

### Using AssetHandlers

Occasionally, especially in notebooks, it's handy to use an AssetHandler on its own. In this case I simply create a handler, and skip the process of setting up a namer.

[Example code here]

## Executors

Parameters are local variables. Early on, the whole configuration was stored as a local variable for each executor. This made it difficult to know what parameters in the configuration were used in an executor. To remedy this, there's a guideline to avoid defining self.cfg = cfg.

Similarly at one point data was loaded in the initialization. This causes many problems - especially that initialization becomes very slow. The exception to this rule is for the instrumentation parameters; we often load those at the start, in order to vet that settings for the instrument do not conflict. <!-- This may need to change. I've considered making the instrumentation as a global object but this also scares me for when multiprocessing is needed. -->

We've looked at some of the parameters while setting up your local system. Now we'll look at some of the Python objects that use them.

Each step -- pulling in some data and parameters and spitting out some different data -- occurs within an Executor. The Executor has a few basic things that it must do:
- define data Assets
- define parameters
- define the procedure

### Setting Up an Executor

When creating a new stage of a pipeline, I first sketch out:
- What's output
- What's input
- What needs to be done

I then put together the pipeline stage in the yaml.

my_stage:
  assets_out:
    my_output:
      handler: OutputHandler
      path_template: "{root}/{dataset}/{stage}/{split}/{sim}/filename.ext"
  assets_in:
    my_input1: {stage: some_stage}
    my_input2: {stage: some_stage}
  dir_name: "MyStage"
  splits: [train, valid, test]

Then I create a python module and create a class. I must have

from cmbml.core import BaseStageExecutor

class SomeExecutor(BaseStageExecutor):
  def __init__(self, cfg):
    super().__init__(cfg, stage_str='my_stage')

    self.out_asset = self.assets_out['my_output']

    self.in_1 = self.assets_in['my_input1']
    self.in_2 = self.assets_in['my_input2']

The stage_str in the call to the super class must match the pipeline stage name in the yaml ("my_stage"). Similarly, the keys for the self.assets_out and self.assets_in must match the asset labels in the yaml.

My __init__ function usually also needs several parameters from the config, like

    self.param1 = cfg.param1
    self.param2 = cfg.model.param2

Now my inputs and outputs are defined.

I suggest also adding tags for the handlers to be used. This does two things:
1. I can quickly reference the handler when working later
2. I'm sure that the handler has been imported and is available to the module

Last, I may need to use some data throughout an executor. If this is the case, within the __init__() function, I define that instance variable and assign it to None so that I know it's there.

Next I need to set up the Executor to actually do something. An "execute()" function must be defined. Most executors iterate through several simulations - like a preprocessing executor. I try to load things, especially maps, as infrequently as possible. If I'm using the same mask on many simulations, I load it at the top of the executor. I usually also have lines for the logger so that I'm able to follow along in the console and my logfiles.

I may need to iterate through many splits and many simulations. In this case, `execute()` has a loop of the form

for split in self.splits:
    with self.name_tracker.set_context("split", split.name):
        self.process_split(split)

This shows one of the ways I interact with the Namer class. I've set the context that, while all processing is done on the split, when any asset gets a path, the split has the correct name.

That pattern of execute() functions is so common that the BaseStageExecutor class has a default_execute() which can be used instead, e.g.

def execute(self) -> None:
    logger.debug(f"Running {self.__class__.__name__} execute()")
    self.default_execute()

My `process_split()`s often have the same structure, of setting up a loop, setting up the context, then calling the process_sim function.

Further granularity is possible. In some cases, I must handle all the detector frequencies for the observation data.

The Namer automatically handles the root, src_root, dataset, and stage portions of filename templates, so no context is needed for those.

### Alternative Executor Structures

There are a couple different structures I've found useful. Iterating over all simulations can be slow. Simple processes are better set up for multiprocessing (e.g., [statistic gathering](cmbml/demo_patch_nn/stage_executors/C_find_dataset_stats_parallel.py)). Alternatively, when training models, I use a PyTorch DataLoader to handle iteration (e.g., [training](cmbml/demo_patch_nn/stage_executors/E_train.py)).

In both those cases, I use the mechanics of the Namer to build sets of references. I don't use the asset.read() function directly, but instead use the asset.path property to get the path, and refer to an appropriate handler.

### More on the Stage Yaml

In addition to assets_out, assets_in, splits, and dir_name, there are some optional tags that can help. See [this README](cfg/pipeline/README.md) for more details.

## Splits

The split helper class is straightforward. It contains some convenience functions for iterating over the contained simulations. The splits are set up by the BaseStageExecutor and stored as a list. I can then simply iterate over the splits, as `for split in self.splits:`. The split object has how many elements to iterate over, from the splits.yaml.

## LogMaker

This is created in your top level script. This helps with traceability by backing up all scripts and configurations. Originally, we were looking at a pipeline centered around DVC, but this seemed to be a more flexible solution. It will duplicate all modules and configuration files and store them in the stage's output directory. In the pipeline yaml, setting make_stage_log: False will disable this. It does create many small files; check and remove these periodically if you're doing many edits and reruns.

A few cautions relating to these logs:
- We've had to set aside three keywords: 'basic', 'default', and 'null'. The LogMaker will ignore configuration yamls with these names.
- External libraries are not backed up. Changes to those will not be tracked.

### Using LogMaker

In the top level script, include these two lines at the start:

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

You will also need to pass the logmaker instance to the PipelineContext.

And this line at the end:
    log_maker.copy_hydra_run_to_dataset_log()

## PipelineContext

The PipelineContext manages how the pipeline is run and sets the order of Executors. It has a prerun_pipeline() method that serves for debugging the configuration yamls (checking for missing parameters before beginning a long series of processes). It also has a run_pipeline() method that creates and executes each stage.

### Using PipelineContext

In the top level script, you'll need to create an instance of the class, then add each stage. Once all stages are added, call the prerun_pipeline() method to verify each Executor can be loaded. Then call the run_pipeline() method. I recommend enclosing the last one in a try: block in order to store the logs of any failure.

The structure of a top level script should therefore be something like:

import stuff

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def run_cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # Add stages
    pipeline_context.add_pipe(FirstStageExecutor)
    pipeline_context.add_pipe(SecondStageExecutor)

    pipeline_context.prerun_pipeline()

    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


# Conclusion

This was a lot! It was more than I expected. At this point the essential elements have been described. I'll give more concrete examples in the next couple notebooks, showing how I use all these elements together.