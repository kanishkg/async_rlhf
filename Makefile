ORG := snow
USER := $(shell eai account get --field name --no-header)
ACCOUNT := mnoukhov
JOB_ACCOUNT ?= $(ORG).$(ACCOUNT) # snow_infiniband.pyllmd

NOW := $(shell date +%s)

GITHUB_REPO := git@github.com:mnoukhov/trl_summarize.git
REPO_NAME := trl_summarize
WANDB_API_KEY_FILE := /home/toolkit/wandb_api_key
DONT_CLONE_FOLDER := results
WANDB_ENTITY := mila-language-drift
WANDB_PROJECT := trl

IMAGE := registry.console.elementai.com/snow.interactive_toolkit/default
IMAGE_REVISION ?= latest

CPU ?= 2
CPU_MEM ?= 64
GPU ?= 1
GPU_MEM ?= 32
# Another option "Tesla T4". You will need `GPU=4` to use that. 
# GPU_TYPE ?= "A100"
MAX_RUN_TIME ?= 7200

# Accelerate settings
NPROC ?= 1
FP ?= fp16

# RAND_ID := $(shell python -c 'import random; print(random.randint(0, int(1e9)))')
DATE_ID := $(shell date +%Y_%m_%d__%H_%M_%S)
JOB_NAME ?= $(REPO_NAME)_$(DATE_ID)
OLD_JOB_NAME := $(JOB_NAME)_old_${NOW}
WANDB_API_KEY := $(shell cat ${WANDB_API_KEY_FILE})

# a function to create a git snapshot based on the active commit
# usage: $(call create_snapshot)
define create_snapshot
	@( [ -d $(_WORKDIR) ] && \
	cd $(_WORKDIR) && \
	[ `git rev-parse HEAD` = $(REVISION) ] && \
	[ -z "`git status -s`" ] && \
	echo "using existing snapshot" ) \
	|| \
	( echo "creating a new snapshot ..." && \
	rm -rf $(_WORKDIR) && \
	git clone --filter=blob:none --no-checkout $(GITHUB_REPO) $(_WORKDIR) && \
	cd $(_WORKDIR) && \
	git sparse-checkout init && \
	git sparse-checkout set --no-cone '/*' '!$(DONT_CLONE_FOLDER)' && \
	git checkout $(REVISION) && \
	pwd && \
	echo "snapshot created successfully")
endef

# a function to rename a job
# usage: $(call rename_job,account,old_name,new_name)
define rename_job
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id --state all`;\
	[ -z $${id} ] || eai job set $${id} --name $(3)
endef

# a function to kill a job
# usage: $(call kill_job,account,name)
define kill_job
	@id=`eai job ls --account $(1) -N $(2) -n1 --field id --state alive`;\
	[ -z $${id} ] || (\
		eai job kill `eai job ls --account $(1) -N $(2) -n1 --field id`;\
		while [ `eai job ls --account $(1) -N $(2) -n1 --field state` != "CANCELLED" ];\
		do\
			echo "Waiting for your old job to cancel...";\
			sleep 5; \
		done;\
	)
endef

# a function to wait for a job to be RUNNING
# usage: $(call wait_for_job_running,account,name)
define wait_for_job_running
	@id=`eai job ls --account $(1) -N $(2) -n1 --field id --state all`;\
	[ -z $${id} ] || (\
		while [ `eai job ls --account $(1) -N $(2) -n1 --field state` != "RUNNING" ];\
		do\
			echo "Waiting for your job to start...";\
			sleep 5; \
		done;\
		echo "Job running as $${id}";\
	)
endef


# a function to get information about a job
# usage: $(call get_job_info,account,name)
define get_job_info
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id`;\
	[ -z $${id} ] && echo "No job found" || (\
		eai job info $${id};\
	)
endef

# a function to get the logs of a job
# usage: $(call get_job_logs,account,name)
define get_job_logs
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id`;\
	[ -z $${id} ] && echo "No job found" || (\
		eai job logs -f $${id};\
	)
endef

REVISION ?= $(shell git rev-parse HEAD)
SNAPSHOT ?= 1
ifeq ($(SNAPSHOT),1)
	_WORKDIR := /home/toolkit/snapshots/$(REVISION)
else
	_WORKDIR := $(PWD)
endif

LOCAL ?= 0  # Use LOCAL=1 for local execution of the latest snapshot
DRY_RUN ?= 0  # Use DRY_RUN=1 to print the commands instead of executing them
# define DRY_RUN_PREFIX
ifeq ($(DRY_RUN), 1)
	_DRY_RUN_PREFIX := @echo '
	_DRY_RUN_SUFFIX := '
else
	_DRY_RUN_PREFIX :=
	_DRY_RUN_SUFFIX :=
endif
_RED=\033[0;31m
_NO_COLOR=\033[0m

# `make job` will use the Conda executable that CONDA_EXE points to.
# If you have a Conda environment activated,
# `make job` will use this environment for the launched job.
ENV ?= $(CONDA_DEFAULT_ENV)
# By default we launch the job in a Conda environment
CONDA ?= 1
# Optionally we launch the job using Huggingface Accelerate
ACCELERATE ?= 0
# Optionally we launch the job using Deepspeed (integrated in Huggingface Accelerate)
DEEPSPEED ?= 0
# To avoid users relying on a local configuration generated by `accelerate config`,
# we provide a default accelerate configuration file,
# which internally calls a deepspeed configuration file
ACCELERATE_CFG ?= conf/deepspeed/accelerate_base.yaml
ACCELERATE_LOCAL_CFG ?= conf/deepspeed/accelerate_local.yaml
DEEPSPEED_CFG ?= configs/deepspeed_zero2.yaml

_CONDA_PREFIX := $(CONDA_EXE) run -n $(ENV) --no-capture-output
ifeq ($(NPROC), 1)
	_ACCELERATE_PREFIX := accelerate launch --mixed_precision=$(FP) --config_file $(ACCELERATE_LOCAL_CFG)
else
	_ACCELERATE_PREFIX := accelerate launch --multi_gpu --mixed_precision=$(FP) --num_processes $(NPROC)
endif
_DEEPSPEED_PREFIX := accelerate launch --config_file $(DEEPSPEED_CFG) --mixed_precision=$(FP) --num_processes $(NPROC) 

_COMMAND := bash -c "$(COMMAND)"
_CONDA_COMMAND := $(_CONDA_PREFIX) bash -c "$(COMMAND)"
_ACCELERATE_COMMAND := $(_CONDA_PREFIX) $(_ACCELERATE_PREFIX) $(COMMAND)
_DEEPSPEED_COMMAND := $(_CONDA_PREFIX) $(_DEEPSPEED_PREFIX) $(COMMAND)

_CPU := $(CPU)
_CPU_MEM := $(CPU_MEM)
_GPU := $(GPU)

ifeq ($(CONDA), 1)
	_COMMAND := $(_CONDA_COMMAND)
endif
ifeq ($(ACCELERATE), 1)
	_COMMAND := $(_ACCELERATE_COMMAND)
	_CPU := $$(($(NPROC) * $(CPU)))
	_CPU_MEM := $$(($(NPROC) * $(CPU_MEM)))
	_GPU := $$(($(NPROC) * $(GPU)))
endif
ifeq ($(DEEPSPEED), 1)
	_COMMAND := $(_DEEPSPEED_COMMAND)
	_CPU := $$(($(NPROC) * $(CPU)))
	_CPU_MEM := $$(($(NPROC) * $(CPU_MEM)))
	_GPU := $$(($(NPROC) * $(GPU)))
endif

# _PYTHONPATH := $(_WORKDIR):$(_WORKDIR)/llmd2-core/src:$(_WORKDIR)/llmd2-finetune/src:$(_WORKDIR)/llmd2-integration/src
_PYTHONPATH := 

.PHONY: job
job:
ifndef COMMAND
	@echo "Must specify the command to run"
	exit 1
endif
ifeq ($(DEEPSPEED), 1)
ifeq ($(ACCELERATE), 1)
	printf "${_RED} ERROR: ACCELERATE=1 incompatible with DEEPSPEED=1! ${_NO_COLOR}\n"
	exit 1
endif
endif
ifeq ($(SNAPSHOT), 1)
	$(call create_snapshot)
else
ifneq ($(LOCAL),1)
	printf "${_RED} WARNING: strongly consider SNAPSHOT=1 for launching remote jobs! ${_NO_COLOR}\n"
endif
endif
ifeq ($(LOCAL), 1)
	$(_DRY_RUN_PREFIX) cd $(_WORKDIR) && \
	PYTHONPATH=${_PYTHONPATH} $(_COMMAND) $(_DRY_RUN_SUFFIX)
else
	$(call rename_job,$(ACCOUNT),$(JOB_NAME),$(OLD_JOB_NAME))
	$(_DRY_RUN_PREFIX) eai job submit \
		--name $(JOB_NAME) \
		--account $(JOB_ACCOUNT) \
		--env HOME=/home/toolkit \
		--env PYTHONPATH=${_PYTHONPATH} \
		--env HF_HOME=${HF_HOME} \
		--env WANDB_API_KEY=${WANDB_API_KEY} \
		--env WANDB_ENTITY=${WANDB_ENTITY} \
		--env WANDB_PROJECT=${WANDB_PROJECT} \
		--workdir $(_WORKDIR) \
		--image $(IMAGE):$(IMAGE_REVISION) \
		--data $(ORG).$(USER).home:/home/toolkit \
		--cpu $(_CPU) \
		--mem $(_CPU_MEM) \
		--gpu $(_GPU) \
		--gpu-mem $(GPU_MEM) \
		--restartable \
		-- $(_COMMAND) $(_DRY_RUN_SUFFIX)
endif
