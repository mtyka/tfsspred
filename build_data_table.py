"""Reads PDBs, HHMs and sequences and builds input data.
"""
import sys
import os
import math
import md5
from collections import defaultdict
import numpy as np
import argparse

dist_stats = dict()
for i in range(0,20):
 dist_stats[i] = []


dssp_label = {
 ' ': 0,
 'H': 1,
 'E': 2,
 'T': 3,
 'S': 4,
 'G': 5,
 'B': 6,
 'I': 7,
}

RESIDUE_3LETTER_TO_1LETTER = {
    '---': ':',
  'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
  'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
  'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
  'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
  'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

RESIDUE_1LETTER_TO_3_LETTER = {
    '-':'---',
    'A':'ALA',
    'R':'ARG',
    'N':'ASN',
    'D':'ASP',
    'C':'CYS',
    'E':'GLU',
    'Q':'GLN',
    'G':'GLY',
    'H':'HIS',
    'I':'ILE',
    'L':'LEU',
    'K':'LYS',
    'M':'MET',
    'F':'PHE',
    'P':'PRO',
    'S':'SER',
    'T':'THR',
    'W':'TRP',
    'Y':'TYR',
    'V':'VAL'
}

RESIDUE_1LETTER_TO_ENCODING = {
    '-':0,
    'A':1,
    'R':2,
    'N':3,
    'D':4,
    'C':5,
    'E':6,
    'Q':7,
    'G':8,
    'H':9,
    'I':10,
    'L':11,
    'K':12,
    'M':13,
    'F':14,
    'P':15,
    'S':16,
    'T':17,
    'W':18,
    'Y':19,
    'V':20
}

def CalcSqrDistance(i, j):
  r12 = np.subtract(i, j)
  return np.inner(r12,r12);

def CalcTorsion(i, a, b, j):
  r12 = np.subtract(i, a)
  r23 = np.subtract(a, b)
  r34 = np.subtract(b, j)
  A = np.cross(r12, r23)
  B = np.cross(r23, r34)
  C = np.cross(r23, A)
  rA = math.sqrt(np.inner(A,A));
  rB = math.sqrt(np.inner(B,B));
  rC = math.sqrt(np.inner(C,C));
  cos_phi = np.inner(A, B) / (rA * rB);
  sin_phi = np.inner(C, B) / (rC * rB);
  return -math.atan2(sin_phi, cos_phi);

class PDB(object):
  def __init__(self):
    self.res = {}
    pass

  def ReadFromFile(self, filename):
    for line in open(filename).xreadlines():
      if line[0:5] == "ATOM ":
        new_atom = {}
        new_atom["num"] = int(line[6:11].strip())
        new_atom["name"] = line[12:16].strip()
        new_atom["resname"] = line[17:20].strip()
        new_atom["resname_1"] = RESIDUE_3LETTER_TO_1LETTER[new_atom["resname"]]
        new_atom["chain"] = line[21].strip()
        new_atom["resnum"] = int(line[22:26].strip())
        new_atom["x"] = float(line[30:38])
        new_atom["y"] = float(line[38:46])
        new_atom["z"] = float(line[46:54])
        new_atom["occ"] = float(line[54:60])
        new_atom["temp"] = float(line[60:66])
        new_atom["line"] = line
        if not new_atom["resnum"] in self.res: self.res[new_atom["resnum"]] = {}
        if not "atoms" in self.res[new_atom["resnum"]]: self.res[new_atom["resnum"]]["atoms"] = {}
        self.res[new_atom["resnum"]]["atoms"][new_atom["name"]] = new_atom
        self.res[new_atom["resnum"]]["resname"] = new_atom["resname"]
        self.res[new_atom["resnum"]]["resname_1"] = new_atom["resname_1"]
        self.res[new_atom["resnum"]]["chain"] = new_atom["chain"]

  def ReadDSSPFile(self, filename):
    ready = False
    for line in open(filename).xreadlines():
      if not ready:
        if "RESIDUE AA STRUCTURE BP1 BP2" in line:
          ready = True
        continue
      try: resnum = int(line[5:10])
      except: continue
      resletter = line[13]
      if resletter not in "ACDEFGHIKLMNPQRSTVWY":
        continue

      ss = line[16]
      bp1 = int(line[25:29])
      bp2 = int(line[29:33])
      acc = int(line[34:38])
      if resnum in self.res:
        if self.res[resnum]["resname_1"] != resletter:
          print "Residue Mismatch in ", self.res[resnum], resletter
        self.res[resnum]["dssp"] = ss
        self.res[resnum]["acc"] = acc

  def ReadHHMFile(self, filename):
    hhm_aas = ['A','C','D','E','F','G','H','I','K','L',
               'M','N','P','Q','R','S','T','V','W','Y']
    hhm_tran = ['M->M','M->I','M->D','I->M','I->I','D->M',
                'D->D','Neff','Neff_I','Neff_D']
    hhm = {}
    ready=False
    lastresnum = -1
    for line in open(filename).xreadlines():
      if len(line) < 7: continue
      if not ready:
        if line[0:7] == "NULL   ":
          ready = True
        continue

      if line[0:7] != '       ':
        resletter = line[0]
        try: resnum = int(line[2:7])
        except: continue
        data = line[7:]
        tokens = data.split()
        if len(tokens) != 21:
          print "Syntax ERROR!", len(tokens), tokens
          sys.exit(1)
        hhm[resnum] = {}
        lastresnum = resnum
        for i,t in enumerate(tokens[0:20]):
          hhm[resnum][hhm_aas[i]] = 2**(float(t)/-1000.0) if t != '*' else 0.0
      else:
        if lastresnum < 0:
          continue
        data = line[7:]
        tokens = data.split()
        if len(tokens) != 10:
          print "Syntax ERROR2!", len(tokens), tokens
          sys.exit(1)
        for i,t in enumerate(tokens):
          if hhm_tran[i].startswith('Neff'):
            hhm[lastresnum][hhm_tran[i]] = t
          else:
            hhm[lastresnum][hhm_tran[i]] = 2**(float(t)/-1000.0) if t != '*' else 0.0

    for ir in self.res:
      if "atoms" not in self.res[ir]: continue
      self.res[ir]["hhm"] = hhm[ir]
      self.res[ir]["hhmfeatures"] = []
      for i in hhm_aas:
        self.res[ir]["hhmfeatures"].append(hhm[ir][i])
      for i in hhm_tran[0:6]:
        self.res[ir]["hhmfeatures"].append(hhm[ir][i])

  def GetAtomTorsion(self,A,B,C,D):
    return CalcTorsion(
        [A["x"], A["y"], A["z"]],
        [B["x"], B["y"], B["z"]],
        [C["x"], C["y"], C["z"]],
        [D["x"], D["y"], D["z"]])

  def GetAtomSqrDistance(self,A,B):
    return CalcSqrDistance(
        [A["x"], A["y"], A["z"]],
        [B["x"], B["y"], B["z"]])

  def GetPsi(self, resnum):
    N  = self.res[resnum]["atoms"]["N"]
    CA = self.res[resnum]["atoms"]["CA"]
    C  = self.res[resnum]["atoms"]["C"]
    N2  = self.res[resnum+1]["atoms"]["N"]
    return self.GetAtomTorsion(N,CA,C,N2)

  def GetPhi(self, resnum):
    C  = self.res[resnum-1]["atoms"]["C"]
    N  = self.res[resnum]["atoms"]["N"]
    CA = self.res[resnum]["atoms"]["CA"]
    C2  = self.res[resnum]["atoms"]["C"]
    return self.GetAtomTorsion(C,N,CA,C2)

  def GetOmega(self, resnum):
    CA = self.res[resnum-1]["atoms"]["CA"]
    C  = self.res[resnum-1]["atoms"]["C"]
    N  = self.res[resnum]["atoms"]["N"]
    CA2 = self.res[resnum]["atoms"]["CA"]
    return self.GetAtomTorsion(CA,C,N,CA2)

  def GetPhiPsi(self, k):
    phi = self.GetPhi(k)
    psi = self.GetPsi(k)
    return phi, psi

  def GetPhiPsiTransformed(self, k):
    try:
      phi = self.GetPhi(k)
      sphi = math.sin(phi)
      cphi = math.cos(phi)
    except:
      phi = 0
      sphi = 0
      cphi = 0

    try:
      psi = self.GetPsi(k)
      spsi = math.sin(psi)
      cpsi = math.cos(psi)
    except:
      psi = 0
      spsi = 0
      cpsi = 0
    return phi, psi, sphi, cphi, spsi, cpsi

  def GetContext(self, res, margin):
    context = ""
    for i in range(res-margin, res+margin+1):
      try:
         context += self.res[i]["resname_1"]
      except:
         context += '-'
    return context


  def GetLabels(self, res, margin, clusters, add_omega_class=False):
    labels = []
    class_offset = 0
    for i in range(res-margin, res+margin+1):
      try:
        phi = self.GetPhi(i)
        psi = self.GetPsi(i)
        cl = FindClass(phi,psi,clusters)
        labels.append(class_offset+cl)
      except: pass
      class_offset += len(clusters)
    return labels

  def PrintPhiPsis(self):
    torus_major_radius = 1
    torus_minor_radius = 0.5
    for k in self.res:
      try:
        context = self.GetContext(k, 6)
        psi = self.GetPsi(k)
        phi = self.GetPhi(k)
        omega = self.GetOmega(k)
        omega_shift = omega + math.pi/2.0
        if omega_shift>math.pi: omega_shift-=2*math.pi
        phi_shift = phi
        if phi>0: phi_shift-=2*math.pi
        phi_shift+=math.pi
        cx = torus_major_radius * math.cos(psi)
        cy = torus_major_radius * math.sin(psi)
        tx = (torus_major_radius + torus_minor_radius *  math.cos(phi))* math.cos(psi);
        ty = (torus_major_radius + torus_minor_radius *  math.cos(phi))* math.sin(psi);
        tz = torus_minor_radius *  math.sin(phi)

        phi_trans = 0.5*(phi_shift+math.pi)
        sx = torus_major_radius * math.cos(psi) * math.sin(phi_trans)
        sy = torus_major_radius * math.sin(psi) * math.sin(phi_trans)
        sz = torus_major_radius * math.cos(phi_trans)

        #print context, phi, psi, omega_shift, phi_shift, cx, cy, tx,ty,tz, sx, sy, sz, math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi)
        print context, phi, psi, omega_shift, math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi)
      except:
        pass

  def CalcLocalConf(self, length, atnam="CA", limit = 10):
    medians = [ 0, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 19, 20, 20, 20, 20, 20]

    for i in range(1,len(self.res)):
      rowstr = ""
      code = 0
      for di,j in enumerate(range(i-17, i-2) + range(i+3, i+18)):
        try:
          at_i = self.res[i]["atoms"][atnam]
          at_j = self.res[j]["atoms"][atnam]
        except:
          rowstr += "-"
          continue
        sqrdist = self.GetAtomSqrDistance(at_i, at_j)
        dist = math.sqrt(sqrdist)
        diff = abs(i-j)
        limit = 20
        if diff < len(medians):
          limit = medians[diff]
          dist_stats[diff].append(dist)
          #if int(dist*4) < len(dist_stats[diff]):
          #  dist_stats[diff][int(dist*4 )] += 1

        if dist < limit:
          code |= 1<<di
          rowstr += "*" #str(int(dist))
        else:
          rowstr += "-"
      #print rowstr, code
      self.res[i]["localconf"] = code 

  def GetSequenceFeature(self, padding):
    first = min(self.res.keys())
    last = max(self.res.keys())
    sequence = []
    for i, ir in enumerate(range( first-padding,  last+padding+1)):
      if ir in self.res: sequence.append(RESIDUE_1LETTER_TO_ENCODING[self.res[ir]["resname_1"]])
      else:              sequence.append(0)
    return np.array(sequence, dtype=np.int8)

  def GetHMMFeature(self, padding):
    first = min(self.res.keys())
    last = max(self.res.keys())
    hmm = []
    for i, ir in enumerate(range( first-padding,  last+padding+1)):
      try: feature_list = np.array(self.res[ir]["hhmfeatures"])
      except: feature_list = np.array([0.0]*26)
      hmm.append(feature_list)
    return np.vstack(hmm).astype(np.float16)

  def GetDSSPFeature(self, padding):
    first = min(self.res.keys())
    last = max(self.res.keys())
    dssp = []
    for i, ir in enumerate(range( first-padding,  last+padding+1)):
      try: dssp.append(dssp_label[self.res[ir]["dssp"]])
      except: dssp.append(0)
    return np.array(dssp, dtype=np.int8)
  
  def GetACCFeature(self, padding):
    first = min(self.res.keys())
    last = max(self.res.keys())
    acc = []
    for i, ir in enumerate(range( first-padding,  last+padding+1)):
      try: acc.append(acc_label[self.res[ir]["acc"]])
      except: acc.append(0)
    return np.array(acc, dtype=np.int8)
  
  def GetLocalconfFeature(self, padding):
    first = min(self.res.keys())
    last = max(self.res.keys())
    localconf = []
    for i, ir in enumerate(range( first-padding,  last+padding+1)):
      try: localconf.append(localconf_label[self.res[ir]["localconf"]])
      except: localconf.append(0)
    return np.array(localconf, dtype=np.int8)

def main():
  parser = argparse.ArgumentParser(
      description='Reads PDBs, HHMs and sequences and builds input data.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--input_list', required=True, type=str,
                      help='Input list for PDBs to read.')
  args = parser.parse_args()

  input_files = open(args.input_list).xreadlines()

  all_sequence = []
  all_hhm = []
  all_dssp = []
  all_acc = []
  all_localconf = []

  padding = 20

  for i, input_file in enumerate(input_files):
    input_file = input_file.strip()
    pdb = PDB()
    try: pdb.ReadFromFile(input_file)
    except: pass
    pdb.ReadDSSPFile(input_file + ".dssp")
    pdb.ReadHHMFile(input_file + ".fas.hhm")
    pdb.CalcLocalConf(1)
    
    sequence = pdb.GetSequenceFeature(padding)
    hhm = pdb.GetHMMFeature(padding)
    dssp = pdb.GetDSSPFeature(padding)
    acc = pdb.GetACCFeature(padding)
    localconf = pdb.GetLocalconfFeature(padding)
    assert sequence.shape[0] == hhm.shape[0]
    assert sequence.shape[0] == dssp.shape[0]
    assert sequence.shape[0] == acc.shape[0]
    assert sequence.shape[0] == localconf.shape[0]
    all_sequence.append(sequence)
    all_hhm.append(hhm)
    all_dssp.append(dssp)
    all_acc.append(acc)
    all_localconf.append(localconf)
    if i%100 == 0 : print "Read:", i

#  # print median distances
#  for i in range(0,20):
#    if len(dist_stats[i]) > 0:
#      print "Med: ", i," : ", sorted(dist_stats[i])[len(dist_stats[i])//4]
#    #print "Med: ", i," : ", " ".join([str(i) for i in dist_stats[i]])

  np.savez_compressed("compacter",
      sequence=np.hstack(all_sequence),
      hhm=np.vstack(all_hhm),
      dssp=np.hstack(all_dssp),
      acc=np.vstack(all_acc),
      localconf=np.hstack(all_localconf))

  print "Done generating ", np.hstack(all_sequence).shape[0], " labelled examples"

if __name__ == '__main__':
  main()



