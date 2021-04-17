#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lintao
# Run detector simulation and digitization together

import os
import Sniper
import time

def get_parser():
  import argparse

  nexoTopDir = os.environ.get('NEXOTOP','')

  parser = argparse.ArgumentParser(description='Run nEXO Digitization Simulation.')
  parser.add_argument("--evtmax", type=int, default=-1, help='events to be processed')
  parser.add_argument("--seed", type=int, default=42, help='seed')
  parser.add_argument("--padsize", type=float, default=6., help='Pad Pitch')
  parser.add_argument("--efield", type=float, default=380., help='Electric Field (V/cm)')
  parser.add_argument("--type", default="bb0n", help='Signal type')
  parser.add_argument("--digioutput", default="../data/bb0n-digi.root", help="specify output filename")
  parser.add_argument("--tilemap", default=nexoTopDir+"/nexo-offline/data/tilesMap_6mm.txt", help='specify tiles Map filename')
  parser.add_argument("--localmap", default=nexoTopDir+"/nexo-offline/data/localChannelsMap_6mm.txt", help='specify local Channels Map filename')
  parser.add_argument("--wpFile", default=nexoTopDir+"/nexo-offline/data/singleStripWP6mm.root", help='specify single Pad WP filename')
  parser.add_argument("--noiselib", default=nexoTopDir+"/nexo-offline/data/noise_lib_100e.root", help='specify single Pad WP filename')
  parser.add_argument("--eleclife", type=float, default=10.0e3, help='Electron Life Time(us)')
  parser.add_argument("--sampling", type=float, default=0.5, help='Sampling Inteval')
  parser.add_argument("--induc", type=bool, default=True, help='Induction Sim')
  parser.add_argument("--swf", type=bool, default=True, help='Save Waveform')
  parser.add_argument("--oversampleratio", type=int, default=50, help='oversampling ratio')
  parser.add_argument("--coType", default="Pads", help='weight potential type')
  parser.add_argument("--run", default="run_gamma.in", help="specify run.mac")
  parser.add_argument("--skipEThreshold", type=float, default=700., help='Deposited energy threshold in keV to skip an event')
  return parser


if __name__ == "__main__":
  start_time = time.time()

  parser = get_parser()
  args = parser.parse_args()
  print(args)

  Sniper.setLogLevel(9)
  task = Sniper.Task("task")
  task.setEvtMax(args.evtmax)

  # = random svc =
  import RandomSvc
  rndm = task.createSvc("RandomSvc")
  rndm.property("Seed").set(args.seed)

  # = buffer =
  import BufferMemMgr
  bufMgr = task.createSvc("BufferMemMgr")
  bufMgr.property("TimeWindow").set([0, 0]);

  # = output =
  ros = task.createSvc("RootOutputSvc/OutputSvc")
  ros.property("OutputStreams").set({"/Event/Sim": args.digioutput, "/Event/Elec": args.digioutput})

  # = geometry service =
  import Geometry
  simgeomsvc = task.createSvc("SimGeomSvc")

  # = detsim =
  Sniper.loadDll("libnEXOSim.so")
  g4svc = task.createSvc("G4Svc")
#
  factory = task.createSvc("nEXOSimFactorySvc")
#
  detsimalg = task.createAlg("DetSimAlg")
  detsimalg.property("DetFactory").set(factory.objName())
  detsimalg.property("RunMac").set(args.run)

  # = filter events by deposited energy =
  Sniper.loadDll("libFilterAlg.so")
  mcfilteralg = task.createAlg("McFilterAlg")
  # energy cut in keV
  mcfilteralg.property("energyCut").set(args.skipEThreshold)

  # = digitizer =
  Sniper.loadDll("libChargeDigitizer.so")

  elecsimalg = task.createAlg("ChargeDigitizerAlg")

  #elecsimalg.property("InputFileName").set(args.geom_input)
  elecsimalg.property("tileMapName").set(args.tilemap)
  elecsimalg.property("padsMapName").set(args.localmap)
  elecsimalg.property("wpFileName").set(args.wpFile)
  elecsimalg.property("ElectronLifeT").set(args.eleclife)
  elecsimalg.property("PadSize").set(args.padsize)
  elecsimalg.property("SamplingInteval").set(args.sampling)
  elecsimalg.property("InductionSim").set(args.induc)
  elecsimalg.property("ElectricField").set(args.efield)
  elecsimalg.property("SaveWaveform").set(args.swf)
  elecsimalg.property("Type").set(args.type)
  elecsimalg.property("noiselibfile").set(args.noiselib)
  elecsimalg.property("OverSamplingRatio").set(args.oversampleratio)
  elecsimalg.property("coType").set(args.coType)

  task.show()
  task.run()

  time_elapsed = time.time()-start_time
  print('Simulation finished in {:4.4} min'.format(time_elapsed/60.))

