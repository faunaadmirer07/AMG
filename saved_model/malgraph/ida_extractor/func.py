# coding:utf-8
#
# Reference Lister
#
# List all functions and all references to them in the current section.
#
# Implemented with the idautils module
#
from idaapi import *
from idautils import *
from idc import *
import idc
import networkx as nx
import cfg_constructor as cfg
import cPickle as pickle
import pdb
import time
from raw_graphs import *
from discovRe import *
import jsonlines

def write_data_to_filename(filename, data):
    # data = json.dumps(data)
    with jsonlines.open(filename, mode='a') as writer:
        writer.write(data)
		
def get_seg_list():
	result = []
	total_seg_number = get_segm_qty()
	for n in range(total_seg_number):
		seg = getnseg(n)
		ea = seg.startEA
		seg_type = segtype(ea)
		if seg_type in [1, 3, 7, 8, 9]:
			continue
		result.append(seg)
	return result


def get_func_cfgs_c(start_time):
	binary_name = GetInputFile()
	raw_cfgs = raw_graphs(binary_name)
	externs_eas, ea_externs = processpltSegs()
	seg_list = get_seg_list()
	flag = False
	i = 0
	for segm in seg_list:
		for funcea in Functions(segm.startEA, segm.endEA):
			
			funcname = get_unified_funcname(funcea)
			
			func = get_func(funcea)

			icfg = cfg.getCfg(func, externs_eas, ea_externs)
			
			i = i + 1
			end_time = int(time.time())
			func_f = get_discoverRe_feature(funcea, func, icfg)
			blocks = [(hex(v.startEA), hex(v.endEA)) for v in FlowChart(func)]
			Insts = getIntrs(func)
			if func_f is None:
				flag = True
				break
			end_time = int(time.time())
			if end_time - start_time >= 300:
				flag = True
				break
			raw_g = raw_graph(funcname, icfg, func_f)
			raw_cfgs.append(raw_g)
			
		if flag is True:
			break
	
	return raw_cfgs, flag


def get_unified_funcname(ea):
	funcname = GetFunctionName(ea)
	return funcname

def processpltSegs():
	funcdata = {}
	datafunc = {}
	for n in xrange(get_segm_qty()):
		seg = getnseg(n)
		ea = seg.startEA
		segname = SegName(ea)
		if segname in ['.plt', 'extern', '.MIPS.stubs']:
			start = seg.startEA
			end = seg.endEA
			cur = start
			while cur < end:
				name = get_unified_funcname(cur)
				funcdata[name] = hex(cur)
				datafunc[cur] = name
				cur = NextHead(cur)
	return funcdata, datafunc
