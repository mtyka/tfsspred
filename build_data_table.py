"""Reads PDBs, HHMs and sequences and builds input data.
"""
import sys
import os
import math
import md5
from collections import defaultdict
import numpy as np 
import argparse

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
      if resnum in self.res:
        if self.res[resnum]["resname_1"] != resletter:
          print "Residue Mismatch in ", self.res[resnum], resletter
        self.res[resnum]["dssp"] = ss

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

  def GetDistanceMatrix(self, res_start, length, atnam="CA"):
    for i in range(res_start, res_start+length):
      rowstr = ""
      for j in range(res_start, i):
        try:
          at_i = self.res[i]["atoms"][atnam]
          at_j = self.res[j]["atoms"][atnam]
        except:
          continue
        sqrdist = self.GetAtomSqrDistance(at_i, at_j)
        dist = math.log(math.sqrt(sqrdist)*2, 2)*3-6
        if dist < 10:
          rowstr += str(int(dist))
        else:
          rowstr += "-"
      print rowstr


  def GenerateTrainingExamples(self, input_file):
    input_window = 10
    label_window =  1
   
    n_examples = len(self.res) 
    examples = np.zeros((n_examples, input_window*2 + 1, len(RESIDUE_1LETTER_TO_ENCODING) + 26), dtype=np.float32)
    labels = np.zeros((n_examples, label_window*2 + 1, len(dssp_label)), dtype=np.float32)
     
    for ith_example, k in enumerate(self.res):
      row = ""
      context = self.GetContext(k, input_window)

      ## Generate sequence encoding

      sequence = np.array([RESIDUE_1LETTER_TO_ENCODING[cr] for cr in context], dtype=np.uint8)
      sequence_one_hot = (np.arange(len(RESIDUE_1LETTER_TO_ENCODING)) == sequence[:, None]).astype(np.float32)
      row += context + " " 

      hmm_features = np.zeros((len(context), 26))
      for i, ir in enumerate(range( k-input_window,  k+input_window+1)):
        feature_list = []
        try: feature_list = self.res[ir]["hhmfeatures"]
        except: feature_list = [0.0]*26
        assert len(feature_list) == 26
        hmm_features[i,:] = feature_list
        for f in feature_list: 
          row += "%f "%f

      dssp_features = np.zeros(label_window*2 + 1, dtype=np.uint8)
      for i, ir in enumerate(range( k-label_window,  k+label_window+1)):
        try: dssp_features[i] = dssp_label[self.res[ir]["dssp"]] 
        except: pass 
      dssp_features_one_hot = (np.arange(len(dssp_label)) == dssp_features[:, None]).astype(np.float32)
      for f in dssp_features: 
        row += "%d "%f

      examples[ith_example] = np.hstack((sequence_one_hot, hmm_features))
      labels[ith_example] = dssp_features_one_hot
    return examples, labels

def main():
  parser = argparse.ArgumentParser(
      description='Reads PDBs, HHMs and sequences and builds input data.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--input_list', required=True, type=str,
                      help='Input list for PDBs to read.')
  args = parser.parse_args()

  input_files = open(args.input_list).xreadlines()

  all_examples = [] 
  all_labels = []

  for input_file in input_files:
    input_file = input_file.strip()
    print "Reading: ", input_file
    pdb = PDB()
    try: pdb.ReadFromFile(input_file)
    except: pass
    pdb.ReadDSSPFile(input_file + ".dssp")
    pdb.ReadHHMFile(input_file + ".fas.hhm")
    examples, labels = pdb.GenerateTrainingExamples(input_file)
    all_examples.append(examples)
    all_labels.append(labels)

  np.savez_compressed("all_examples", examples=np.vstack(all_examples), labels=np.vstack(all_labels))
  print "Done generating ", np.vstack(all_examples).shape[0], " labelled examples"

if __name__ == '__main__':
  main()



