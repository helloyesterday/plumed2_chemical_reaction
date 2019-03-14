/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The ves-code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of ves-code, version 1.

   ves-code is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ves-code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with ves-code.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "TargetDistribution.h"
#include "GridIntegrationWeights.h"
#include "VesBias.h"
#include "core/ActionRegister.h"
#include "tools/Grid.h"
#include <iostream>
#include "tools/FileBase.h"
#include "Optimizer.h"
#include "LinearBasisSetExpansion.h"
#include "VesLinearExpansion.h"
#include "tools/Communicator.h"

namespace PLMD {
namespace ves {

//+PLUMEDOC VES_TARGETDIST TD_LORENTZIAN
/*
Lorentzian-tempered target distribution (dynamic).

The frequency of performing this update needs to be set in the
optimizer used in the calculation. Normally it is sufficient
to do it every 100-1000 scale update iterations.

\par Examples
Employ a well-tempered target distribution with a scale factor of 10
\plumedfile
td_welltemp: TD_WELLTEMPERED BIASFACTOR=10
\endplumedfile

*/
//+ENDPLUMEDOC

class TD_Lorentzian: public TargetDistribution {
private:
  unsigned int iter;
  Grid* grad_fes_grid_pntr_ ;
  Grid* norm_bias_ ;
  double scale_factor_;
  int pstride;
  std::vector<double> weight_factor_;
  std::vector<double> weight_2;
//  std::vector<unsigned> nbins;
  void calculatelogTargetDistCoeff();
//  void calculateBiasGradient();
  void calculateGradient(); // returns a grid s.t. sum (gradient w.r.t. arg)^2 are points
  std::vector<unsigned int> nbasisf;
  unsigned int ncoeffs;
  std::vector<BasisFunctions*> pntrs_to_basisf_ ;
   LinearBasisSetExpansion* pntr_to_bias_expansion_;
   CoeffsVector* log_target_coeffs_pntr_;
   CoeffsVector* fes_coeffs_pntr_;
   size_t stride;
   size_t rank;
   std::vector<double> bf_norm;
   std::string gradient_init_filename_;
//   Communicator comm_in;
 public:
  static void registerKeywords(Keywords&);
  explicit TD_Lorentzian(const ActionOptions& ao);
  void updateGrid();
  double getValue(const std::vector<double>&) const;
  ~TD_Lorentzian() {}
};


PLUMED_REGISTER_ACTION(TD_Lorentzian,"TD_LORENTZIAN")


void TD_Lorentzian::registerKeywords(Keywords& keys) {
  TargetDistribution::registerKeywords(keys);
  keys.add("compulsory","SCALEFACTOR","The scale factor used for the Lorentzian distribution.");
  keys.add("compulsory","WEIGHTFACTOR","The scale factor used for the Lorentzian distribution.");
  keys.add("optional","PRINT_STRIDE","The frequency of printing the gradient of the FES");
}


TD_Lorentzian::TD_Lorentzian(const ActionOptions& ao):
  PLUMED_VES_TARGETDISTRIBUTION_INIT(ao),
  iter(0),
  grad_fes_grid_pntr_(NULL),
  norm_bias_(NULL),
  scale_factor_(0.0),
  pstride(0),
  weight_factor_(0.0),
  weight_2(0.0),
//  nbins(0),
  nbasisf(0.0),
  ncoeffs(0),
  pntrs_to_basisf_(0.0),
  pntr_to_bias_expansion_(NULL),
  log_target_coeffs_pntr_(NULL),
  fes_coeffs_pntr_(NULL),
  stride(1),
  rank(0),
  bf_norm(1.0)
//  comm_in(NULL)
{
  parse("SCALEFACTOR",scale_factor_);
  parseVector("WEIGHTFACTOR",weight_factor_);
  if(keywords.exists("PRINT_STRIDE")) {
    parse("PRINT_STRIDE",pstride);
  }

//  if(keywords.exists("GRID_BINS")) {
//  std::cout<<"here1"<<std::endl;
//    parseVector("GRID_BINS",nbins);
//  std::cout<<"here2"<<std::endl;
//  }
  setDynamic();
  setFesGridNeeded();
  checkRead();
}


double TD_Lorentzian::getValue(const std::vector<double>& argument) const {
  plumed_merror("getValue not implemented for TD_Lorentzian");
  return 0.0;
}


void TD_Lorentzian::updateGrid() {
  if(pntr_to_bias_expansion_ == NULL){
    VesBias* vesbias = getPntrToVesBias();
    VesLinearExpansion* veslinear = static_cast<VesLinearExpansion*>(vesbias);
    if(veslinear) {
      pntrs_to_basisf_ = veslinear->get_basisf_pntrs();
      pntr_to_bias_expansion_ = veslinear->get_bias_expansion_pntr();
    }
    else{
      plumed_merror("Use Ves_Linear_Expansion type for VesBias when the target distribution is Lorentzian");
    }
    nbasisf.assign(pntr_to_bias_expansion_->getNumberOfArguments(),0.0);
    weight_2.assign(pntr_to_bias_expansion_->getNumberOfArguments(),0.0);
    ncoeffs=pntr_to_bias_expansion_->getNumberOfCoeffs();
    if(weight_factor_.size()!=0){
      plumed_massert(weight_factor_.size()==pntr_to_bias_expansion_->getNumberOfArguments(),"ERROR: number of WEIGHT_FACTOR must match number of arguments");
    }
    for(unsigned int i=0; i<pntr_to_bias_expansion_->getNumberOfArguments(); i++){
      nbasisf[i]=pntr_to_bias_expansion_->getNumberOfBasisFunctions()[i];
      if(pntrs_to_basisf_[i]->getType()!="trigonometric_cos-sin" and pntrs_to_basisf_[i]->getType()!="Legendre"){
        plumed_merror("Only BF_Fourier or BF_LEGENDRE as of now");
      }
      if(pntrs_to_basisf_[i]->getType()=="trigonometric_cos-sin"){
        bf_norm[i]=(pntrs_to_basisf_[i]->intervalRange())/2.0;
      }
      else{
        bf_norm[i]=1.0;
      }
     weight_2[i]=weight_factor_[i]*weight_factor_[i];
     }
     gradient_init_filename_="gradFES."+vesbias->getLabel()+".";
 }

  Communicator& comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();
  stride=comm_in.Get_size();
  rank=comm_in.Get_rank();
  double scale = scale_factor_ * scale_factor_;
//  double beta_2 = getBeta()*getBeta();
//  double weight_factor_2 = weight_factor_ * weight_factor_;
  plumed_massert(getFesGridPntr()!=NULL,"the FES grid has to be linked to use TD_Lorentzian!");
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getTargetDistGridPntr());
  double norm = 0.0;
  calculateGradient();
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
//    double value = weight_factor_2*beta_2*(grad_fes_grid_pntr_->getValue(l));
    double value = grad_fes_grid_pntr_->getValue(l);
    value = scale + value;
    value = scale_factor_ /value;
//    value = weight_factor_* value;
    targetDistGrid().setValue(l,value);
    norm += integration_weights[l]*value;
    value = -std::log(value);
    logTargetDistGrid().setValue(l,value);
  }
  targetDistGrid().scaleAllValuesAndDerivatives(1.0/norm);
  calculatelogTargetDistCoeff();
  logTargetDistGrid().setMinToZero();
}

//void TD_Lorentzian::calculateGradient(){
//  std::cout<<"here3"<<std::endl;
//  iter= iter+1;
//  std::vector<double> dx_ = getFesGridPntr()->getDx();
//  std::vector<unsigned int> nbins =pntr_to_bias_expansion_->getGridBins(); //getFesGridPntr()->getNbin();
//  std::vector<bool> isperiodic =getFesGridPntr()->getIsPeriodic(); 
//  if(grad_fes_grid_pntr_==NULL){
//    std::vector<std::string> min = getFesGridPntr()->getMin();
//    std::vector<std::string> max = getFesGridPntr()->getMax();
//    std::vector<std::string> args = getFesGridPntr()->getArgNames();
////    for(unsigned int i; i<args.size();i++){
////      if(isperiodic[i]==0){
////        nbins[i]=nbins[i]-1;
////      }
////    }
//    grad_fes_grid_pntr_ = new Grid("grad_fes",args,min,max,nbins,false,false,true,isperiodic,min,max);
//    std::cout<<"here"<<std::endl;
//    std::cout<<"nbins "<<grad_fes_grid_pntr_->getNbin()[0]<<std::endl;
//    std::cout<<"dx "<<grad_fes_grid_pntr_->getDx()[0]<<std::endl;
//    std::cout<<"min "<<grad_fes_grid_pntr_->getMin()[0]<<std::endl;
//    std::cout<<"max "<<grad_fes_grid_pntr_->getMax()[0]<<std::endl;
//    std::cout<<"isperiodic "<<grad_fes_grid_pntr_->getIsPeriodic()[0]<<std::endl;
//
//  }
//  nbins = getFesGridPntr()->getNbin();
//  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
//    std::vector<unsigned int> ind = getFesGridPntr()->getIndices(l);
//    std::vector<unsigned int> ind_c = getFesGridPntr()->getIndices(l);
//    double grad = 0.0;
//    for(unsigned int k=0; k<getFesGridPntr()->getDimension(); k++) {
//        double val1 = 0.0;
//        double val2 = 0.0;
//        double value =0.0;
//        if((ind[k])%nbins[k]==0){
//           if(isperiodic[k]){
//              ind_c[k]=ind[k]+1;
//              val1 = getFesGridPntr()->getValue(ind_c);
//              ind_c[k]=ind[k]+nbins[k]-1;
//              val2 = getFesGridPntr()->getValue(ind_c);
//              value = (val1 - val2)/(2*dx_[k]);  
//            }
//           else{
//              ind_c[k]= ind[k]+1;
//              val1 = getFesGridPntr()->getValue(ind);
//              val2 = getFesGridPntr()->getValue(ind_c);
//              value = (val1 - val2)/dx_[k];
//            }
//        }     
//        else if((ind[k]+1)%nbins[k]==0){
//            if(isperiodic[k]){
//              ind_c[k]=ind[k]-nbins[k]+1;
//              val1 = getFesGridPntr()->getValue(ind_c);
//              ind_c[k]=ind[k]-1;
//              val2 = getFesGridPntr()->getValue(ind_c);
//              value = (val1 - val2)/(2*dx_[k]);
//            }
//            else{
//              ind_c[k] = ind[k]-1;
//              val1 = getFesGridPntr()->getValue(ind);
//              val2 = getFesGridPntr()->getValue(ind_c);
//              value = (val2 - val1)/dx_[k];
//            }
//        }
//        else{
//            ind_c[k]=ind[k]+1;
//            val1 = getFesGridPntr()->getValue(ind_c);
//            ind_c[k]=ind[k]-1;
//            val2 = getFesGridPntr()->getValue(ind_c);
//            value = (val1 - val2)/(2*dx_[k]);
//        }
//        grad+= (value*value);
//    }
//    grad_fes_grid_pntr_->setValue(l,grad);
// }
//  std::string gradFes_fname = "gradFes."+std::to_string(iter)+".data" ;
//  OFile ofile;
//  ofile.link(*this);
//  ofile.enforceBackup();
//  ofile.open(gradFes_fname);
//  grad_fes_grid_pntr_->writeToFile(ofile);
//  ofile.close();
//
//}
//
//void TD_Lorentzian::calculateBiasGradient(){
//  iter= iter+1;
//  std::vector<double> dx_ = getFesGridPntr()->getDx();
//  std::vector<unsigned> nbins =pntr_to_bias_expansion_->getGridBins(); // getFesGridPntr()->getNbin();
//  std::vector<bool> isperiodic = getFesGridPntr()->getIsPeriodic();
//  if(grad_fes_grid_pntr_==NULL){
//    std::vector<std::string> min = getFesGridPntr()->getMin();
//    std::vector<std::string> max = getFesGridPntr()->getMax();
//    std::vector<std::string> args = getFesGridPntr()->getArgNames();
////    std::vector<bool> isperiodic = getFesGridPntr()->getIsPeriodic();
//    grad_fes_grid_pntr_ = new Grid("grad_fes",args,min,max,nbins,false,false,true,isperiodic,min,max);
//    norm_bias_ = new Grid("norm-bias",args,min,max,nbins,false,false,true,isperiodic,min,max);
//  }
//  
////bias without cutoff grid normalized and minima set to zero
//  double minimum=0.0;
//  double normb=0.0;
//  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
//   double value = -getBiasWithoutCutoffGridPntr()->getValue(l);
//   if(value<minimum){
//     minimum=value;
//   }
//    std::cout<<"minimum "<<minimum<< "iter " <<iter<<std::endl;
//  }
//  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {                                          
//    double value = getBiasWithoutCutoffGridPntr()->getValue(l);                                       
//    value =( value - minimum);
//    normb = normb + value;                                                                               
//    norm_bias_->setValue(l,-value);
////    std::cout<<"norm "<<norm << "iter " << iter <<std::endl;
////    std::cout<<"norm"<<norm<<std::endl;
//  } 
//    std::cout<<"norm "<<normb << "iter " << iter <<std::endl;
////    std::cout<<"minimum "<<minimum << "iter " << iter <<std::endl;
////gradient calculation
//  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
//    std::vector<unsigned int> ind = getFesGridPntr()->getIndices(l);
//    std::vector<unsigned int> ind_c = getFesGridPntr()->getIndices(l);
//    double grad = 0.0;
//    for(unsigned int k=0; k<getFesGridPntr()->getDimension(); k++) {
//        double val1 = 0.0;
//        double val2 = 0.0;
//        double value =0.0;
//        if((ind[k])%nbins[k]==0){
//           if(isperiodic[k]){
//              ind_c[k]=ind[k]+1;
//              val1 = norm_bias_->getValue(ind_c);
//              ind_c[k]=ind[k]+nbins[k]-1;
//              val2 = norm_bias_->getValue(ind_c);
//              if(normb!=0.0){
//                val1 = val1/normb;
//                val2 = val2/normb;
//              }
//              value = (val1 - val2)/(2*dx_[k]);  
//            }
//           else{
//              ind_c[k]= ind[k]+1;
//              val1 = norm_bias_->getValue(ind);
//              val2 = norm_bias_->getValue(ind_c);
//              if(normb!=0.0){
//                val1 = val1/normb;
//                val2 = val2/normb;
//              }
//              value = (val1 - val2)/(dx_[k]);
//            }
//        }     
//        else if((ind[k]+1)%nbins[k]==0){
//            if(isperiodic[k]){
//              ind_c[k]=ind[k]-nbins[k]+1;
//              val1 = norm_bias_->getValue(ind_c);
//              ind_c[k]=ind[k]-1;
//              val2 = norm_bias_->getValue(ind_c);
//              if(normb!=0.0){
//                val1 = val1/normb;
//                val2 = val2/normb;
//              }
//              value = (val1 - val2)/(2*dx_[k]);
//            }
//            else{
//              ind_c[k]= ind[k]+1;
//              val1 = norm_bias_->getValue(ind);
//              val2 = norm_bias_->getValue(ind_c);
//              if(normb!=0.0){
//                val1 = val1/normb;
//                val2 = val2/normb;
//              }
//              value = (val1 - val2)/(dx_[k]);
//            }
//        }
//        else{
//            ind_c[k]=ind[k]+1;
//            val1 = norm_bias_->getValue(ind_c);
//            ind_c[k]=ind[k]-1;
//            val2 = norm_bias_->getValue(ind_c);
//            if(normb!=0.0){
//              val1 = val1/normb;
//              val2 = val2/normb;
//            }
//            value = (val1 - val2)/(2*dx_[k]);
//        }
//        grad+= (value*value);
//    }
//    grad_fes_grid_pntr_->setValue(l,grad);
// }
//  std::string gradFes_fname = "gradFes."+std::to_string(iter)+".data" ;
//  OFile ofile;
//  ofile.link(*this);
//  ofile.enforceBackup();
//  ofile.open(gradFes_fname);
//  grad_fes_grid_pntr_->writeToFile(ofile);
//  ofile.close();
//
//}


void TD_Lorentzian::calculatelogTargetDistCoeff(){
//  if(log_target_coeffs_pntr_ == NULL){
//    std::vector<Value*> args_pntrs_= pntr_to_bias_expansion_->getPntrsToArguments();
//    Communicator& comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();
//    log_target_coeffs_pntr_ = new CoeffsVector("logTarget.coeffs",args_pntrs_,pntrs_to_basisf_,*comm_in,true);
//  }
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getLogTargetDistGridPntr());
  std::vector<double> coeff;
  coeff.assign(ncoeffs,0.0);
  std::vector<double> product(nbasisf[0],0.0);
  for(Grid::index_t l=0; l<logTargetDistGrid().getSize(); l++) {
    std::vector<double> args =(logTargetDistGrid().getPoint(l));
    double logTarget = logTargetDistGrid().getValue(l);
    logTarget= integration_weights[l]*(logTarget);
    unsigned int nargs = args.size();
    std::vector<double> args_values_trsfrm(nargs);
    std::vector< std::vector <double> > basisf(nargs);
    std::vector< std::vector <double> > derivs(nargs);
    bool all_inside;
    for(unsigned int k=0; k<nargs; k++) {
      basisf[k].assign(nbasisf[k],0.0);
      derivs[k].assign(nbasisf[k],0.0);
      pntrs_to_basisf_[k]->getAllValues(args[k],args_values_trsfrm[k],all_inside,basisf[k],derivs[k]);
    }
//    for(unsigned int j=0; j<nbasisf[0]; j++){
//      product[j]+=integration_weights[l];
//      std::cout<<"l "<< l<<" j "<< j<<" "<<basisf[0][j]<<std::endl;
//    }
//    size_t stride=1;
//    size_t rank=0;
//    if(comm_in!=NULL){
//      stride=comm_in->Get_size();
//      rank=comm_in->Get_rank();
//    }
    for(size_t i=rank; i<ncoeffs; i+=stride){
      std::vector<unsigned int> indices=log_target_coeffs_pntr_->getIndices(i);
      double basis_curr = 1.0;
      for(unsigned int k=0; k<nargs; k++) {
        basis_curr*= basisf[k][indices[k]]/bf_norm[k];
      }
      coeff[i]+=((logTarget)*basis_curr);
    }
  }
//  std::cout<<"end of basis"<<std::endl;
//  for(unsigned int j=0; j<nbasisf[0]; j++){
//    std::cout<<product[0]<<std::endl;
//  }
//  std::cout<<"end of 1st gradient"<<std::endl;
  for(unsigned int i=rank; i<ncoeffs; i+=stride){
    log_target_coeffs_pntr_->setValue(i,coeff[i]);
  }
}
void TD_Lorentzian::calculateGradient(){                                                 
   Communicator& comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();

   iter= iter+1;
   if(grad_fes_grid_pntr_==NULL){ 
     std::vector<double> dx_ = getFesGridPntr()->getDx();
     std::vector<unsigned int> nbins =pntr_to_bias_expansion_->getGridBins(); // getFesGridPntr()->getNbin();                                        
     std::vector<std::string> min = getFesGridPntr()->getMin();                                        
     std::vector<std::string> max = getFesGridPntr()->getMax();
     std::vector<std::string> args = getFesGridPntr()->getArgNames();
     std::vector<bool> isperiodic = getFesGridPntr()->getIsPeriodic();
     grad_fes_grid_pntr_ = new Grid("grad_fes",args,min,max,nbins,false,false,true,isperiodic,min,max);
   }

   if(fes_coeffs_pntr_ == NULL){
     std::vector<Value*> args_pntrs_= pntr_to_bias_expansion_->getPntrsToArguments();
//     fes_coeffs_pntr_ = new CoeffsVector(pntr_to_bias_expansion_->BiasCoeffs());
//     *comm_in = pntr_to_bias_expansion_->getPntrToBiasCoeffs()->getCommunicator();
     fes_coeffs_pntr_ = new CoeffsVector("logTarget.coeffs",args_pntrs_,pntrs_to_basisf_,comm_in,true);
   }

   if(log_target_coeffs_pntr_ == NULL){
     std::vector<Value*> args_pntrs_= pntr_to_bias_expansion_->getPntrsToArguments();
//     log_target_coeffs_pntr_ = new CoeffsVector(pntr_to_bias_expansion_->BiasCoeffs());
     log_target_coeffs_pntr_ = new CoeffsVector("logTarget.coeffs",args_pntrs_,pntrs_to_basisf_,comm_in,true);
     log_target_coeffs_pntr_->setAllValuesToZero();
//     for(unsigned int k=0; k<pntr_to_bias_expansion_->getNumberOfArguments(); k++){
//       bf_norm[k]=(pntrs_to_basisf_[k]->intervalRange())/2.0;
//     }
   }

   CoeffsVector* bias_coeffs_pntr = pntr_to_bias_expansion_->getPntrToBiasCoeffs();
   for(unsigned int i=0; i<ncoeffs; i++){
     double value = (pntr_to_bias_expansion_->getKbT())*(log_target_coeffs_pntr_->getValue(i));
     value = value - bias_coeffs_pntr->getValue(i);
     fes_coeffs_pntr_->setValue(i,value);
   }
   std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getFesGridPntr());
   double beta_2 =getBeta()*getBeta();
   for(Grid::index_t l=0; l<getFesGridPntr()->getSize(); l++) {
     std::vector<double> args = getFesGridPntr()->getPoint(l);
     unsigned int nargs = args.size();
     std::vector<double> args_values_trsfrm(nargs);
     std::vector< std::vector <double> > basisf(nargs);
     std::vector< std::vector <double> > derivs(nargs);
     bool all_inside = true;
     for(unsigned int k=0; k<nargs; k++) {
       basisf[k].assign(nbasisf[k],0.0);
       derivs[k].assign(nbasisf[k],0.0);
       pntrs_to_basisf_[k]->getAllValues(args[k],args_values_trsfrm[k],all_inside,basisf[k],derivs[k]);
     }
//     for(unsigned int j=0; j<nbasisf[0]; j++){
//       product[j]+=basisf[0][1]*basisf[0][j];
//     }
     std::vector<double> grad_fes_value;
     grad_fes_value.assign(nargs,0.0);
     std::vector<double> derivs_curr;
//     size_t stride=1;
//     size_t rank=0;
//     if(comm_in!=NULL){
//       stride=comm_in->Get_size();
//       rank=comm_in->Get_rank();
//     }
     for(size_t i=rank; i<ncoeffs; i+=stride){
       std::vector<unsigned int> indices=fes_coeffs_pntr_->getIndices(i);
       double coeff = fes_coeffs_pntr_->getValue(i);
       for(unsigned int k=0; k<nargs; k++) {
         derivs_curr.assign(nargs,1.0);
	 for(unsigned int j=0; j<nargs; j++) {
	   if(j!=k){
	     derivs_curr[k]*=basisf[j][indices[j]];
	   }
	 else{
	   derivs_curr[k]*=derivs[j][indices[j]];
	 }
       }
       grad_fes_value[k]+=coeff*derivs_curr[k];
     }
    }
    if(stride!=1){
      for(unsigned int k=0; k<nargs; k++) {
        comm_in.Sum(grad_fes_value[k]);
      }
    }
    double value = 0.0;
    for(unsigned int k=0; k<nargs; k++) {
      value+=beta_2*weight_2[k]*(grad_fes_value[k])*(grad_fes_value[k]);
    }
    grad_fes_grid_pntr_->setValue(l,value);
   }
   std::string time;
   Tools::convert(getPntrToVesBias()->getIterationFilenameSuffix(),time);
//   std::string gradFes_fname = "gradFes."+std::to_string(iter)+".data" ;
   if(pstride!= 0 && getPntrToVesBias()->getIterationCounter()%pstride==0){
     std::string gradFes_fname = gradient_init_filename_+time+".data";
     OFile ofile;
     ofile.link(*this);
     ofile.enforceBackup(); 
     ofile.open(gradFes_fname);
     grad_fes_grid_pntr_->writeToFile(ofile);
     ofile.close();
   }

}


}
}
