??0
?'?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.22v2.8.2-0-g2ea19cbb5758??/
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?u *%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?u *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:  *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name43894*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_33593*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?u *,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?u *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?u *,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?u *
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?u*
dtype0*??
value??B???uBarticleBpageB	wikipediaBwouldBpleaseBoneBlikeBtalkBseeBalsoBthinkBknowBpeopleBeditBuseBarticlesBmayBtimeBthanksBgetBmakeBevenBcouldBgoodBinformationBwantBiBwellBsourcesBwayBnameBpagesBfirstBhelpBeditingBnewBdeletionBgoBneedBsourceBthankBsayBsectionBuserBeditsBmadeBmuchBmanyBreallyB
discussionBusedBfuckBdeletedBfindBworkBsinceBaddBpointBlookBrightBsomeoneBimageBreadBtakeByouBbackBstillBtwoBfactBsaidBlinkB	somethingBblockedBgoingBstopBlistBhiBcontentBwithoutBblockBaddedBusBeditorsBremovedBanotherBhistoryBmightBhoweverBwelcomeBplaceBnoteBsureBdoneBneverBputBletBaskBpersonalBreasonBseemsBfeelBbetterBquestionBcommentB	vandalismBpersonBcaseBanythingBusingBbelieveBactuallyBbestBthingsBlinksBsubjectBcommentsBhopeBwikiBpartBkeepBfreeBpolicyBthingBitBnothingBproblemBalreadyBremoveBchangeBwrongBlittleBtryingBmustBanyoneB	copyrightB
understandBeditorBworldBissueBothersBgiveB	questionsBagreeBthoughBratherBlastBmakingBcomeBreliableByearsBlongBsorryB	differentBcontinueBtagBtryB	referenceBmeanBfoundB
referencesBgreatBleaveBspeedyBtextBsaysBgotBhelloBeveryBipBelseBsimplyBoriginalBfairBaddingBsiteBshowBeitherBdeleteBwordBuB	consensusBcheckBstateBenglishBrequestBprobablyBlifeBenoughBcreatedBshitBrevertedBdayBaroundBopinionBfarBviewBnotableByesBwarBadminBusersBcontributionsBmatterBencyclopediaBthoughtBtheBwriteBmaterialByetBpostBleastBaccountBgivenBclearlyBbookBneedsBsupportBmessageBlotBbadBtellBseemBevidenceBrealBexampleBeverBinsteadBimagesBcorrectBclearBcalledBsayingBhateBalwaysB	importantBwebsiteBmaybeBtrueBconsiderBquiteBnumberBniggerB
guidelinesBoldBbitBcriteriaBclaimBmediaBwhetherBfuckingBstatesBperhapsBwrittenBresearchBlanguageBgettingBtermBversionBohBreviewBmakesBtimesBpovB
consideredBrevertBmentionBseveralBsuckBcannotBchangesBideaBetcBwordsBnoticeB	followingBaddressBbasedBlistedBgroupB•B	regardingBcurrentBfactsBcareBtemplateByearBpossibleBmeansBrulesBdontBheyBattackBkindB	statementBwholeBsecondBmainBmanBassB	mentionedBisBincludeBgeneralBstartBissuesBleftBdateBokBtitleBseenBgayBtopBthreeBsuggestBhappyBtopicBcourseBendBcreateBcallBprovideBlessBsenseBthisBbigBredirectBexplainBprojectBmoveBloveBamericanB
notabilityBschoolBappropriateBsentenceBchangedBstartedBdaysBremovingBdieBinfoBknownBincludedBpictureBlolBnextBlookingBstyleB	communityBanswerBmindBawayBsignBrelevantBanywayBfourBinterestBwarningBorderBthBrecentBsummaryBdiscussBpoliciesB
interestedB	includingBattacksBclaimsBableBeveryoneBwroteB	currentlyBneutralBspecificBtakenBlaterBpositionBpublicBwantedBfullB—BappearsBwritingBwithinBperBniceBstuffBfaithBgodBrelatedBliveBlineB	certainlyBnamesBwishBreportBofficialBalthoughBcountryB
everythingB
especiallyB
completelyBlooksBfatBcommonBprettyBprocessB	thereforeBthatBinvolvedBtodayBsingleBlearnBnewsBstayB	accordingBunlessBeditedBcameBleadBwebBdueBhardBaskedBtruthBresponseBfutureB	obviouslyBstupidBpowerBplacedBreadingBrememberBadminsBpostedBfaggotBsandboxBguyBpastBquoteBtalkingBmoronBexactlyBregardsBargumentBworkingBusefulB	paragraphBnoticedBcityBgameBagoB
appreciateBfalseB	publishedBsystemB	politicalBwpBhighBdeletingBadministratorBfiveB
governmentBdisputeB	revertingBreasonsBformB
particularBsimilarBguessBcBunitedBwhateverBproblemsBemailBpartyBbecomeBcheersB	vandalizeBtriedBmajorBsideBruleBprovidedBnationalBstatedBbannedBtookBbritishBtakingBbB	knowledgeBbooksBneededBstatusBfineBuploadedBalongBdoBcertainBexplanationBalmostBpointsBoftenBreplyBlawBcompanyBentireBdescriptionBfollowB	generallyB	otherwiseBawareBentryBsortBrecentlyBtermsBweekBdecideBshowsBsawBpresentBaloneBsoonBbanBguysB
definitionBcontributingBimproveBfileBpieceBcitedBviewsBmusicBusernameBappearBlikelyBinterestingBgoogleBtestBindeedBopenBjewBcitationBsetBwhiteBexternalB	attentionBareaBsimpleBcuntBtoldBproposedBallowedBstoryBfamilyBshortBactualBtheoryBinternetBdisagreeBsmallBmovedBcategoryBobviousBcontactBnonsenseBmembersBvariousBwentBresultBfilmBenjoyBactionsBandBtypeBmrBtagsBcontextBbandBsucksBciteBthusBjobBbiasBconflictBsitesBauthorBhoursBjewsBjohnBautomaticallyBpreviousBproperBtogetherBcreatingB	citationsB
universityBlongerBbitchBworksBimBworkedBdealBadditionBvalidBhappenedBavoidB	seriouslyBgoesBblackBsourcedBrespectBactionBdeathBproofBnpovBhelpfulBonesBacceptedBhumanBbiasedBcomesBsearchBindicateBtildesBpigBexistB	availableBlargeBopinionsBactBacceptBscienceBhandB	necessaryBunblockB	violationBrightsBattemptB
statementsBhellB	criticismBcauseBdickBaccurateB
personallyBsectionsByeahBlivingBtaggedBuponBsexBdebateBmanualBplayBcopyBseriesBafdBfBcallingB
explainingBmultipleBstandardBnoBassumeBwikiprojectBrestBgermanBmonthsBjulyB
historicalBseriousBmeaningB	situationBdetailsBseparateBdoubtBfixBblockingBrecordBheardBreferB	rationaleBaskingBcompleteBbullshitBvideoBmessagesBusuallyBlevelBwhatBproveB
wikipedianBbehaviorBonlineB
contributeBchurchBperiodBnoneBhugeBthirdBdirectlyB
differenceBdataBsouthBlegalBlackBpillarsBgetsBteamBaugustBsupposedBspeakBfishBqualityBproduceBcoupleBfriendBchangingB
apparentlyBrunBmarchBusesBsignificantBearlyBmeantB	incorrectB
disruptiveB	describedBindiaBgreekBerrorBcultureBcloseBmilitaryBamongBspaceBfieldB	countriesBphotoBheadB	inclusionBjewishBpurposeBgaveBtutorialBokayBabuseBtableBjuneBreleaseBboxBprimaryB	anonymousBshallB	characterBbusinessBcontestBoutsideB	argumentsBukBbringBreportedBspecificallyBsockBpicturesBcontrolBwaitBthereBthinkingBcomingBvoteBallowBmemberBtakesBkillBmodernBforceBworthBconcernsBgivingBhereBearlierBdecidedBcasesBfinallyBdearBpBparticularlyBreadersBballsB
absolutelyBhappenBrespondBorganizationBputtingBadministratorsBnumbersBmajorityBjanuaryBdecisionB	sometimesBmeBdeBmistakeBwhyB	discussedBtotallyBmotherBlostBmerelyBhomeBfriendsBtowardsBredBunsignedBxBinappropriateBwomenBchanceB
scientificBwantsBneitherBlinkedBhouseBexceptB	standardsBindependentBentirelyB–B
irrelevantBgroupsBadviceByourBpostingBinternationalBeatBassholeB
populationBfigureBmillionBchildrenBwankerBlookedB
requestingB
individualB
acceptableBmeetBfagBremovalBaprilBunfortunatelyBunderstandingBnobodyBfanBcivilBcourtBfinalBeventsBpostsBgivesBreaderBnorthBreligionBacrossBbarkBlicenseBpressBmenBbarnstarBbotBwatchBcomputerBnamedB	christianBamericaBsoundBsentBhighlyB
experienceBifBfaceBbornBdeadBinBstBpossiblyBexamplesBlightB
verifiableBpoorBgoneBbehindBminorBdogBaBeventBsongBmineBdirectBknowsBdecemberBpxBwarringBpsB	templatesBkingBproperlyB	septemberBisraelBclubBaheadBcoverBaccountsBlistsBoctoberBmapBdiscussionsBnatureBdespiteBserviceBfunBcockBsolidBresultsBmissingBalbumBcreationBformerB
definitelyBeasilyBageBshownBfrenchBnovemberBexpectBeasyBsoundsBhalfBcoolBstandBmoneyBbasisBfeaturedBspecialBaidsBstrongBspeedilyBstudyBshutB
reasonableBnotedBbodyBrB	difficultBcenturyBintroductionBspamBrrBgBvandalB	religiousBgamesBdamnBinfoboxBweeksBsoBfebruaryB
propagandaB	explainedB	unsourcedBpenisBmovementBseeingB
suggestionBmovieBwonderB	guidelineBmeetsB	wonderingBwarningsBhitlerBlocalBcollegeBsomebodyBmomentBmergeBcontributionBtoBrequiredB
experimentBeffortBbasicBaccessBideasBhearBtvBignoreBeBpopularBforwardBwillingBidiotBcontroversialBmonthBliesBgaBracistBdescribeBamountBcontroversyBboyBarchiveBguideBfullyBepisodeB
discussingBfixedBeffectBwaysBrapeBdavidB	basicallyBieBearthBusageBwikipediansBtestsB	concernedBappreciatedBplacesB
nominationBdetailBjesusBboardByorkBaniB	suggestedBvalueBfindingBtotalB	referringBsocialB
importanceBnaziBbushBwordingBnippleBformatB
referencedB	presentedBmiddleBfactualBestablishedBbecameBindianBconceptBrfcBbottomBquickBconsideringBtellingB
charactersBanymoreBbeyondBagreedBborderBremainBleavingBbroughtB	nominatedBarmyB
repeatedlyBaccusationsBfunnyB
categoriesBjokeBholdB
previouslyBmannerBkeptBreleasedBgladBcontributorsBbtwBpaperBclassBadvertisingBargueBtypesB	supportedBquotesBrussianBreceivedBimmediatelyBapplyBsuperBnegativeBthatsBpoliceBchineseBtroubleB	presidentBluckBshowingBwestB	uploadingBbiographiesBbattleBconcernBhesitateBactiveBwidthBpartsBpreventBnBfightB	somewhereB	authorityBbelongBacademicBrealizeBtopicsBexistsBclaimedB
additionalBmostlyB	extremelyBrealityB	behaviourBclaimingBbastardB
ridiculousB	beginningBpracticeBdickheadBseasonB
legitimateBjoinBtacosBknewBhelpmeBethnicBconstructiveBlateB	requestedBproposalBeastBwebsitesBsocietyBinsertBfitBintendedBencyclopedicBcheckingBupdateBwesternBwhoseBnightBairBsendBrepublicBfuckerB
permissionBhappensBgeorgeBselfBcrapBnotesB	hopefullyBnonfreeBeuropeBstatingBrockBnotBstartingBspellingBspanishBrefrainBfailedBroleB BrfaBrevertsBphraseBconfusedBthoughtsBstepB	biographyBweightBprogramBchildBpriorBkilledBturnBmovingB
regardlessBcodeBsharedBartBconfirmB	companiesBstickBblogBsuggestionsBreturnBnearBsonBmarkBuploadBancientBwalesBrefersBpersonsBclickBwikipediahiBplayedBfffaBkeepingBcountBfootballBchooseBcanadaBwarsBphotosBfamiliarBeuropeanBvsBsexualBcampaignBadmitB	secondaryBjapaneseBcutB
connectionB	additionsBviaBwowBloserB	languagesBjamesB
protectionBpositiveBfatherBbotherBincludesB
backgroundBstudentsBofferBawardBalternativeBrandomBraceBnumerousBcentralB	offensiveBdrBupdatedBinstanceBexactBchinaBspeakingBfreedomBpreferBsadBbreakBmetalBenglandBfrontBexpertBreplaceBaccusedB
sockpuppetBapproachBwaterBlogBjpgBcountyBclickingBcitingB
resolutionBregionBtoneBagendaBfaggotsBstronglyBoriginBlistenBeffortsBthreatsBandorBshareBopposeBinputBfairlyBadBlimitedBflagBfeltBminutesB	protectedBgermanyB	improvingB	perfectlyBnowBfollowedBcantBareasBuselessBbesidesBtrulyBthinksBanywhereBsurelyB
misleadingBcheckedBbandsBtrollBreachedBeyesBexperimentingBhelpingBcommonsBofficeBforumBprofessionalBpointedBmagazineBlieBfocusBdevelopmentBsubjectsBactingBenergyBreferredB
concerningBraisedBproductBexistingBarbitrationBseemedBwatchingB	apologizeBmuslimBexcuseBcopyrightedBplusBsomewhatBlowB	educationBtitlesBlandBhonestBnoesBcoverageBislamBmyBhaBfurthermoreByoungBworseBmichaelBlogicBhigherBanybodyBstudiesBrunningBnormalBcontentsBsuspectBchoiceBcleanBshortlyBrequireBdiedBsaveBhomoBreviewedBgonnaBfailBbunchBstarBreflectBempireBcatholicBclosedBtrustBplayingBhitBpromoteBheldBteabagBfireBwasteBspeechBfucksexBabilityBusaBerrorsBmassBfamousBegB
yourselfgoBnorthernBeasierBrequiresBregardBobjectBmattersBfreezerBsomehowBparticipateBlotsBhenceB	elsewhereBcarefulBbuildingBvandalizingBvBrestoredBreplacedBmedicalBirishB	americansBrangeB
harassmentBquicklyBdropBagainBincidentBcontainsBsizeBreportsBhowBdatesBsillyBpaulBkB	criterionBblankBverticalaligntopBsmellsBrudeBlinesB	existenceBstraightBdisputedBcontainBarbcomBboldBforgetBeyeBirelandBhardlyBprotectBcontributorBappearedBprivateB
mainstreamBiiBafraidBsolutionBtownBtheoriesBsirBcoveredB
associatedBassertBverifyBperfectBnationB	committeeBtalkpageB
originallyBonBglobalBrepeatBkingdomBcareerB	providingBblocksBuserpageBmatchB
conclusionBmessBkeepsB	continuedB	addressedBwomanBwarnedBtaggingBsupposeB	sentencesBromanBjBtranslationB
throughoutBplanB
sufficientBcolorB
frequentlyBrobertBmentionsBblueBprovidesBdeserveBlistingBattitudeBattemptsBdisambiguationBinterpretationBpointingBcausedBallBdudeBbeganBasB
stylewidthBpatheticBleagueBentriesBunionB
supportingBfilesB
conspiracyBblatantBwikipediaquestionsB	newspaperB	confirmedBfiguresBwBmistakesB
mentioningBstubBdumbBshotBrewriteBdomainBafricanB	thousandsBletterB	wikimediaBrecordsBfellowBwidelyB	unblockedBnearlyBfillBbelongsBprovenBnamingBidentityB
eventuallyBclarifyBanalysisBwinBhelpsBvisitB	surprisedBmergedBcopiedB	australiaB	recommendBpoliticsBhonestlyBexperiencedB
cocksuckerBbirthBlondonBhealthBdealingBcontraryB
neutralityB	influenceBpaidBfavorB	summariesBcriticalBgreenBspreadBalBtwiceBpeaceBdetailedB	correctlyBproposeBmodelBinvolvementBfilmsBbenefitBthreadBkeyBignorantB
foundationBcenterB
suggestingBsovietBsignificanceBdegreeBpolishBpassBindividualsBvoiceBnativeBtoolB	structureBsoftwareB	reasoningBpushingBsplitB	potentialBminorityBjustifyBreviewsBpartiesBjapanBstreetB	objectiveBturkishBplotB	confusionB
censorshipBpayBdocumentB
consistentBworryBwishesBobamaBmurderBlaBignoredBgreaterB
compromiseBurBdefinedBlocationBlivesBlibraryBlatestBfeedbackBthreatBpoopB	confusingB	agreementBviolateBrussiaBopeningBappliedBhourBdefendBdoubleB	precedingBdesignBspentBsidesBperspectiveBlawsBevilB	complaintBbusyBreadsBplentyBmissedB
impossibleBmetB	intentionBfightingBstandsBquotedB
departmentBbillBmethodBislandBassociationBjournalBexpandBnetworkBforeignB	specifiedBjunB	attackingBstudentBrestoreBparkBopposedB	mediationBgoalB	evolutionBassumingBverifiabilityBsixBndB
californiaBcreditBoddBfallBcrimeBrepeatedBlB	buttsecksBauthorsBalteredBturnedBresponsibleBremainsBreadyBplayersBbeliefBworstBtrackBdisputesBbriefBversionsBroyalBdevelopBbeginBspendBscholarsBdelayBverifiedBgrammarBcallsBrelationshipBwriterBstrangeBbrotherBradioBnaturalBlatterBforB	determineBreachBparentsBmothjerB
literatureBcitiesBbuildBinsultByaB	signatureBledBfederalBcontribsB
attemptingBlordBlengthBforcesBculturalBschoolsBhBstationBwilliamBtrollingB	excellentBanalB	continuesB	carefullyBimagineB
commercialBsickBlabelBfggtBconductBvillageBblpBareBspecifyBgroundBcuriousBgirlBelectionBcivilityByouiBpeerBbloodBunableBhandsBrefBrateBbackgroundcolorfBintroB	apologiesBsongsBnotwatBbrownBallegationsBpussyBdykBplayerBarguingBunnecessaryBmuslimsBreB	correctedBnoobsBbeliefsBsystemsBlyingBequalBcredibleB
violationsBsignedBresolveB	everybodyB	interviewBconversationB	technicalBexpandedBartistBregularBrdBcapitalBnecessarilyBjustBintelligenceBholderBbobBadvanceB	respondedBlogoB
inaccurateBimprovementBimprovedBextremeB	reportingBjusticeB
impressionBfaBtouchBtenBnationsBindustryBbibleBattackedBzeroBtabBliberalBjimboBaccusingBmarBidentifyBforeverBlearningBillegalBexpressBarchivesBplainBoppositeBasideBtendBrequestsBphysicsBlargestB
dictionaryBappliesBpeoplesBlogicalB
historiansBchickenBaverageB
understoodBphysicalB—precedingBfansB	treatmentBprojectsBgenreBepisodesBsevenBdailyBchannelBwikipediaimageBunclearBappealBbecomingB
philosophyBdeskB	terroristBbioBsourcingBpublicationBisraeliBcontributedBlowerBjudgeBdifferencesBactivityBpowersBpeterBoverallBcanadianBknowingBfeelingB	violatingBinvestigationBaccuseBvotesBquitBnonBiranBintentBheartBfoolBensureBarabBwaitingBrevisionBpassedBlivedBhtmlBforgotBcomparedBbalanceBuncivilBislamicB
christiansB
australianB	yesterdayBundoBseekBletsBfriendlyBfranceBengageBbecomesBhundredsBbaseBturkeyBremarksBbrokenBsuitableBroomBjimB
accusationBsuggestsBpureBsakeB	redirectsBoilB
explicitlyB
whatsoeverBpopBbyeBretardedBphilippineslongBcriminalByoutubeB	resourcesBofBmainlyBtomorrowB	terrorismBlargerBjackBitalianBcomplexBapologyBlatinB
constantlyBchristianityBallegedBtreatedB	representBdriveBcouncilBchargeBresolvedBlearnedBheilBdeservesBlinkingBinsideB
televisionBlikesBheBroundB
objectionsBfeatureBcomB	assertionBpersianBownerBaccuracyBwasBfemaleBeasternBdistrictBcreatorBcoupledBpuppetBontoBinstructionsBtoolsBshowedBpurposesBnormallyBfranklyBcanBaspectsB
washingtonBvandalsBshameBkidBforcedB
assistanceBwikipediaarticlesBsockpuppetryB
paragraphsBoughtB
activitiesBunBgrowBformalB	documentsBpushBfebBdamageB
recognizedBongoingBimpactBdarkBwifeBmexicansBitemBpromotionalBkindlyBkidsBfakeBeditionBactsBroadBpickBorganizationsBcarBsupportsBmongoBidBfoodBfollowsBbcBawardsBsubstantialBlettingB	interestsBspeciesBsecurityB
registeredBformsBexistedBserveB	professorB
newsletterBmaleBconstitutionBstoppedBhideBfringeBexpectedBbarBafricaBusualB	encourageBdecBapartBabusingB
respondingBoptionBmaintainBdidntBbuiltBmorningB
comparisonBcleanupBolderBinformBuniqueB
technologyB	developedBunlikeB
officiallyB	insultingBillBdraftBcommonlyBslightlyB
productionBgfdlBbuttonBbritainB	attemptedBstoriesBsmithBriverBrepliedBprogressBfreelyBconstitutesBahBmoreoverBleadingBipsBboobsBextraBconservativeBstalkingB
oppositionBheadingBendedB
criticismsB
vandalizedB
scientistsBracismBloseB
documentedBlargelyBfearBdesignedB
commentaryBworthyBiranianB
collectionBclarificationB
appearanceBrelationBopportunityBgroundsBwhoeverBportalBdirtyB	communistBcloselyB	sincerelyBleaderB	ignoranceBspiritBrichardB	materialsBinsistBinitialBfolksBexplainsB	directionBprimeBignoringBhitsBgrandBdrawBcellpaddingBsouthernBcatBunacceptableB	territoryBskillsBfinishedB	establishBdirectedBbollocksBapparentB	recognizeBlovesBessentiallyB	editorialBwideBspeculationBjanBfuckinBbrainBasianBwprsBtaskBqueenBkoreanBjustificationBwhereasBproducedBfunctionB
formattingBentitledBdenyBuniverseBthemBlackingBcompareBangryB	rewrittenBimpliesBcharlesBtypicalBequallyB
disruptionBchoseBtiredBintendBengagedBsmartBmasterBjonesBtowardBservicesBeffectsBdiffBcupBbenefitsBallowsBofffuckB	financialBcomplainBviolatesBrecallBproBnationalityBeconomicBdareBchartB	addressesBadditionallyBsecretBsanB
reputationBmissBmassiveBgainBfailsB
describingBbiggestBadviseB‎BunknownB	reviewingB	prominentBproductsBcrossBtomBridBcriticsB
continuingBstageBnikkoBgreeceBfeelingsBsecondlyBhaveBforthBhelpedBbbcBarmsBpapersBofferedBharmBdivisionB
determinedB
submissionBpublicationsBetBunconstructiveBsportsBreceiveBpromiseBhopingBfailureBaprB·BtriviaB	operationB	challengeBanonBplanetB	objectionBthomasB
statisticsB	newcomersBlocatedBhotBbloodyBbelievedB
unreliableBunfairBspeedBrecordedBmapsB	holocaustB	happeningBcocksB	candidateBproassadhanibalBmereBhurtBdistinctionBcredibilityBwheneverBuglyB	promotionBpossibilityBniggasBmatchesBtellsBrealizedBraiseBnowhereB	justifiedBcensorBbutBbetBholyBcongratulationsBchatBchapterBabusiveBthreateningBlettersBfastBatheistBviolenceBvaluableBsunB
identifiedBgenocideBengineBsafeBdesireBviolatedBgrantedBfiredBdocumentationBasiaBweakBtechnicallyBsatisfyBintelligentB	indicatesBhandleBexpertsB
democraticBconsiderationBnoticeboardBmeritBliarBhinduBreactionBaboutBtrialBprofileBgreatlyBfeaturesBdefineBcoversBcarryBcaresBanywaysB	scholarlyBledeBitemsBfixingBchrisBbitchesfuckBansweredBthrowB
successfulBsexsexBrenderB	principleBiraqBdemandBsocalledBdoesntBbeatBweBwallBtreatBtraditionalB	specifiesBhistoricB
candidatesByoBriceheyBproseB	positionsBmillionsB
introducedBimplyBelementsB
discoveredBdeneidaccessBcrimesBassumedBaircraftBupsetBqualifyBheadsBbuyBvastBsoldBrefuseBmouthBmarkedBgreatestBannoyingBwritersBupBunitBtargetBscreenBnoticesBcausesBbelievesBtipsBtillBseaBinvitedB	describesBbearBartistsB	relationsBoffendedBkillingB	historianB	harassingBarmenianB	watchlistBstuckBprodBlossBdestroyBconventionsB
assessmentBanimalsBstandingBreplaceableBpmBeraBcoreBchosenBtrainingBpakistanBdeepBconstantBaspectBthreerevertBrefsBmistakenBgottenBdirectorBdeliberatelyBdcBcrazyBcaughtBanswersBfyiBalbumsB	pointlessBjoeBextentBdaBunlikelyBtooBregionalBpolandBmsB	macedoniaBinsertedBidiotsBbankBadministrationBfaultB	directoryBdefenseByoufuckB
terroristsBmachineB	literallyBlickBheavyBgunBguiltyBcheeseBbabyBadministrativeBabsurdBthenB	submittedB	reputableBprovesB	exceptionBbalancedBvotingBpunkB	promotingBmethodsBmadB
conditionsBanimalB	viewpointBpollB
pagedeleteBorganisationBmoviesBministerBlawdyBdiseaseB	bitchfuckBsitBinformedB
conventionB	commentedBcastBserbianBpropertyBlicensedBemBwarnBscaleBpasteBminuteB
macedonianBlimitBlaughBinnocentBgirlsBbringingBbotheredBnotrhbysouthbanofBcensusBvagueBtexasBstaffBpainBfictionBsurveyBrequirementsBmissionB	committedB	christmasBarchivedBamazingBaiB	publisherBprovinceBhomelandBoxymoronBoriginsBmateBhoaxB	expansionBdrinkBclassmainpagebgBchristBsceneBromneyB	respectedB
occupationB	fictionalBdisplayBconclusionsBzBspokenBrepresentedB	preferredBmedicineBinformativeBweirdBwastingBtradeBtonyBtitledB
relativelyBquotingBpoorlyBpicBmemoryBladyBjumpBgasBanthonyBshipBrejectedBpresenceBbridgeBwontBriskB	redundantBprintB	primarilyBmarketB	ethnicityB	deletionsBbyBteamsBsteveBnetBnazisBhighestBedBopenedBnovBmikeBdefinitionsBvanBryanBregardedB	judgementBhiddenBfoxBdogsBcocksuckingBbiggerBbackgroundcolorBalrightBwritesBsalesBracialB
principlesBinlineBcloserB
subjectiveBscholarBrollbackB	relevanceBfacB	extensiveB	defendingBcoldBaddsBacknowledgeBrsBpressureBinsultsB	greetingsBfeelsBapplicationBtreeB	socialistBquestionableBproudBphoneBmixedB	dedicatedBmuhammadBdidBdemonstrateBabsoluteB	selectingBscriptBdecentBboysBtestingBsubstantiallyBstepsBserbiaBmergingBcircumstancesBchesterB
reasonablyBmarriedBmanagedBkoreaBinternalBimoBformedBcommunicationBnotificationBessayBbehalfBwpnpovBtalksBreviewerBpurelyB
newspapersB
managementBleadsBhearingB	expressedBcitesButcBservedB
republicanBoccurredBnominateBmartinBhorseBfoundedBwannaBsuccessBfrankBdubiousBvictimBveryBvandalisingBtommyBpowerfulBexplanationsBalotBscottBscopeB	preciselyBownedB	containedBblameBallowingBvotedBsmallerBwhoBinviteBgreeksB
generationBflightBenterB	effectiveB
complaintsB
supportersBpassageBisntB
equivalentBcriminalwarBvictimsB	ownershipBnationalistBmovesBmentalBmeetingB	decisionsB	bunksteveBblankingBassfuckBwindowsBorthodoxBmamasBfacebookBclimateBbreakingBundueBturksBtrivialBspammingBrelatingB	recogniseBnotheBgoldBvarietyBunitsBtaxBstoneBpatternB	introduceBimprovementsBcoiBbumB	responsesB	procedureBlockedBjoinedBhairB
facilitateBchargesBwingBwikipediawikiprojectBtcBsockpuppetsBmittBjacksonBeggBdoctorBconformanceBcitizensBarabicBwhilstBstartsB	quotationB
intentionsBincreaseBgenderBcroatianBarguedBrefusedBplanningBkhanBfashionBcuntsBsometimeBsisterBresourceBpiecesBinventedBencyclopaediaBcreativeBcontemporaryBclassicBbenBtripBproblematicBgermansBfootnoteBcostBclueBmarriageB	convincedB	connectedBapprovedB
accordanceBauthoritiesBwhatsBsoldiersB
publishingBmaryBheavilyB
everywhereBdiffsB
copyrightsB	beautifulBvideosBprinceBpotentiallyBoffenseBmythBislandsB	electionsBdutchBdanBaudioBafghanistanBadoptedB	similarlyBconformBterribleB	reversionBreturnedBhuhB	democracyBartsBarseBsubmitBloggedBfinishBdeclinedB
assumptionB
accuratelyBsummerBpuppetryBleadersB	involvingBehBdisneyBcatchBtablesBspainB
marcolfuckBendsBachieveBviewedBreferencingBprovedBminimumBdealtBaccordinglyBsupremeBoutBnovelBmusicalBjerkB	destroyedBboymamasBselectedB	instituteBfiledB	worthlessBtypingBrealiseBpoliteBperformanceBperformBoverviewB	integrityBincBfilledB
commentingBacknowledgedBreduceBranB
productiveBintellectualBhesheB
expressionBdropdownBbadlyBarrogantBadmittedB	worldwideBwantingBrenamedB
redirectedBindefinitelyB	factuallyBeuBdeniedBdeclaredBdanceBblindBakaBsymbolBsolveB	remainingBregisterBpublishBisbnBheritageBhallBgeneticBgarbageB
controlledB
challengedB	unrelatedB	traditionBtourBprintedBmuseumBmoreBmassacreBkindsBjosephBfundamentalBfraudBdiBcultBcsdBofficerBobscureBindependenceBciaBchiefBbrothersBtrollsB
subsequentB	searchingB
representsB	permittedBmeasureB	indicatedBholdingBfourthBelementBdaughterBdanielBclosingBchildishBbiographicalBsinghB
penissmallBpendingBpaddingBhahaBgenuineBforgiveBfavourB
correctingB	checkuserBtextsBspeaksBrecreateBrecognitionBpalestinianBmisunderstandingBcounterBconceptsBballB
translatedBprogramsBmailBkosovoBharryBgottaBfounderBenemyB
constituteB
addressingBwpblpBturnsBthousandBplacingBforumsBdaveBtempleB
protectingBpretendBnastyBhumansBeconomyB
attributedBaliveBwweBrequirementBrareBpoliticallyBoptionsBcorrectionsB	conflictsBawesomeBwarmingB	virtuallyB
revolutionBremarkB
homosexualBfactorB	conditionB	completedBchicagoBcausingBcardBcalmBbrokeBtalkedBspeakersBslowBresponsibilityBpumpBpromotedBportionBorphanedBissuedB	contestedB	classicalBcellBcaptionBurlBsecondsB	licensingBfiguredBeditwarringBdocB
correctionBcongressBblogsBarrestB
reversionsBnickBmeantimeBlevelsBgoalsBbackedB	announcedBvomitBunbiasedBtonightBtheyBseekingBoldidBoctB
meaningfulBircBhorribleBconvinceB
conferenceBytmndinBwildBservesBrepresentativeBoccurBnuclearBhebrewBflowBffffffBextendedBentityBbroadBalbanianBvaluesBthreatenBremindBfocusedBenvironmentB	practicesBmergerBitalyBguitarBcooperationBcomprehensiveB	adminshipBsquareBsolelyBsaltBparticipationBfavoriteBbullyBblahBarabsB	apologiseBactorBvolumeBvehicleBupdatingBteacherBspotBskinBrouteB	receivingBobtainedBneverthelessBmoralBlazyBdukeBcapacityBbagBundoingBsoleBrunsBrelativeBpreparedBpassingBleeBlakeBexchangeBcopyvioBbanningB	typicallyBsignificantlyB	regularlyBmeaninglessB	initiallyBfoughtBfackB
disgustingBwiseBwarrantB	translateBregionsBprisonBoxfordBjudaismB
interviewsBgrowingBbrandBtriesB
threatenedBsupplyBseBpissB
persistentBinterventionBiiiBfirmBdrugBdramaBdisagreementB
supposedlyBsuitBsuicideBstubsBrichBpopeBplansBnotionBhillB
encouragedB	communismBcleaningB	centuriesBantisemitismBandrewB	wrestlingB	vandaliseBstatsB	receptionBnooneBmodifiedBhostBfallsBengineeringBelB	displayedB	suspectedBsettledBserverBsecurityfuckBscotlandBrarelyBopposingBgalleryBfieldsBfcBemphasisBcarriedB	bulgarianBbsBassistBstrictlyBsortsBmexicoBmeritsBjudgmentBiveBiceBdvdBdemonstratedBdecadesBcuntbagBbitsBbellBtinyBtacticsBstrikeBlibelBheroBdeeplyBcontentiousBcolumnBsaluteBpettyBideologyBfascistB	dangerousBcommunicateBaimBwhoreBterminologyBrenameB	qualifiedBpdfBparticipantsBjimmyBhenryBcyprusBvaginaBsendingBrubbishBproceedBpermanentlyBloggingBlimitsBengagingB	wonderfulBwindBukraineBtaughtBstyleverticalaligntopBspellBreliabilityBpcBmoonB	identicalB	expertiseBcryBconstructionBbuddyBsportBseperateBriseB	restoringBpriceBkindaB	insertingBheightBdroppedBcollaborationB
assertionsBstylebackgroundcolorfBsellBproxyBpatienceB	operatingBnationalismB
aggressiveByouthB
ultimatelyBtrainBrepresentationBoutcomeBmomBlawyerBhidingByoubollocksB	wellknownBreplacementBplaysBdozenBdistinctBcoastBbasteredbasteredBaccidentBwindowBweekendBteachBpreciseBleavesB
hypothesisBgrantBflagsB	fartchinaBendingBdistanceBdescentBcontrastBcomplicatedBbullyingBwithBstringB
quotationsBqBparisBmidB
indicationBholeBfamiliesBdoorBdescriptionsBdatabaseBcampBavoidedBwinningBsavedB	revisionsBpickedBhmmBheaderBguidanceBcapableBadamBveggietalesBsuddenlyBreflectsB	permanentBpartisanBmattB
invitationB	instancesBhundredBegyptBdenialBcomplainingBcitizenB	scientistBposterBgeorgiaBfredBeffectivelyBconcernthanksB
commissionBcolorsBclearerBaliBunreferencedBsignpostBscumBpersonalityB	performedBinvolveBimdbBassureB
applicableBalterBwalkBverificationBstableBreducedB
presentingBmilesBjournalsBjoiningBholdsBdifferentlyBdatedBdBancestryBagreesB	useredgarBuntrueB	universalB	ukrainianB
stylecolorBsittingB	proposingBoutrightBorangeBmudBgrossBeightBceaseBamongstB	alternateBalexB
subsectionBstanceB	religionsBobjectsBmodeB
incivilityBeducationalBcensoredBbaseballBwporBtravelBmosBhistoricallyBexposedBderivedBbecauseBbannerBancestryfuckoffjewishBamountsBvictoryBsmileBshootingBschemeBringBrenamingB	organizedBorBlikedBflyingBcrisisB	buildingsB	blatantlyBadultBwritingsButterlyBteaBmumBhonorBfromBsingerBpromotesB
presumablyB	palestineBoffB
minoritiesBantisemiticB	alexanderBsocksBrulingB	perceivedBfreshBeraseBdigitalBcopiesBchampionshipBadvancedBwtfBwikipediarequestsBvalleyBstationsBsBharassBfeetBdislikeBbringsBantiBvalidityBsurfaceBstarsBsemiprotectedBpubliclyBpasswordBmotivesBmindsBmeatB
journalistBgangBfirstlyB	expandingBdabBcourtesyBcircumcisionBbrieflyBboxesBturningBspeakerBruledBrelyBrecommendedBpornBphrasesB
photographBpatentB
membershipBhimBhatredBdeclineB	contactedBconfirmationBwizardBwatchedBtownsBloadBlacksB
influencedBhostileBgoldenBcoBcdBbayBwpvB	welcome BurgeBstrictBsinglesBsbsBrationalBmathBenforcementBcomicBcombinedBatBadvocateBacademyBupperBsweetBswedishBsurpriseBshapeB
plagiarismBmarksBheatBfactorsBcontroversiesBconsistencyBanimeBactedByourselfBvisibleBtoeB	synthesisBshannonBserbsBretiredBregimeBfantasyBdocumentaryBdepthBcommunitiesBadvertisementBworkersB	repeatingB
preferenceBmodifyBfloridaBfellBemptyB	economicsBdiegoBcomplyBcancerB→B	temporaryBseesBrevealedBreleasesB
nonnotableB	naturallyBlikewiseBframeBfindsBedwardBcontinuallyBapprovalBandyB	volunteerBspelledB
respectiveB	replacingB
reconsiderBprizeBinvasionBimportantlyBericBdadBcountsBconsistentlyBbeBaffectBabsenceBwaveBrepliesBniggaBlmaoBlabeledBiqB	forgottenBfameBcentreBbradburyBweezerBsuckuBsleepBshootBobservedBlyricsBlickerBenteredB	disagreesBdependsBcontractB	conductedBclassesBaugBattachedB
altogetherBwpaniBweaselB
thirdpartyBteachingBsumBstoreBspendingBsettingBremainedBprBinvolvesBfalselyBexclusivelyBenBemailsBdeemedBcleanedBcellspacingBbuttBadvisedBwtcB
thoroughlyB	substanceBsadlyBnyBmotionBlegallyBjdelanoyBfbiB	essentialBdozensBdeathsBcroatiaBcasteB	arbitraryBairportBwhenBspokeBsilenceBpullBlastlyBkevinB	jerusalemBgratefulB	footnotesBfitsB	duplicateBconsequencesB	appearingBtimelineBthoroughBterritoriesBresultedBplannedBparticipatedB	meanwhileBguruB	excessiveBcowardBbarelyBagesBagentBullmannBsellingBputsB
punishmentBplantBpicsB	motivatedBlibertyBkingsBinstitutionB
incompleteBemailedBdestructionBcompetitionBcommandBbbbBauthoritativeBthrowingB
simplifiedBreverseBphdBparticipatingBnominationsBmisinformationBlayBlatelyBinterpretedBexplicitBexerciseBdoesBdeletedthisBcnnBaudienceBstyleborderBsagetohBprotestBoutlineBnationalisticBjohnsonB	immediateBharderBgenericBdeviceBbirthdayBunusualBtranslationsBtehBsigningBrevisedBrailwayBradicalB
operationsBmonitorBinspiredBincorrectlyBhomosexualityBenemiesBdynastyB	computersBcomicsBcolourBclintonB	clarifiedBburnB	ambiguousBactorsBwikiloveBvillagesBunfortunateBspringBshitfuckBseniorBpleasureBpatientBobservationBlockB
illustrateBconfuseBalternativesBaffectedBactivelyBtroopsBsurroundingBresearchersBreformB	qualifiesBprovingBpronunciationBphotographsB
phenomenonB	marketingB	incidentsBhandledBelectedBdreamBachievedBscholarshipBislesBheckBentertainmentB	countlessB	albaniansB	advantageBwhereBtoughBsolvedB	practicalBnavyBironBinterpretationsBgoodbyeBcodesBchampionBcentraliststupidBweaponsBvirginBunreasonableBsyriaBsubpageB	sanctionsBrootB	rewritingBownsBordinaryBnotingBlosB
incrediblyBhisherBhaventB	userspaceBstudioBstudiedBsetsBroseBreplyingBrejectB
privilegesBpathBkurdsBdrugsBdrawnBdevotedB	corporateBcopyingBchuckBbranchB
allegationBwikipediaadministratorsB	usernhrhsBuserboxBturkicBsuperiorBstephenBrevealBracesBpuppetsB
psychologyBprideBjobsBinvalidBinsultedBimplyingBhospitalBflyBdavisBcoveringB	convictedBbedByellowB
viewpointsBtribesBtribeB
situationsBorderedB
noteworthyBmmBjayBharshBenforceBdegreesBcreatesBchurchesBblowBagencyB
vandalisedBsightBrootsB
motivationB	microsoftBlaidB	infoboxesBianBgradeBfloorBdefinesBcabalBassumptionsBalternativelyBstalinBsignsBromanianB
recognisedBpickingBoccupiedBnicknameBlosingBeverydayBeveningB	dependingBcuzBchartsBbizarreBsepBsamB	retrievedB
parliamentB	locationsBlengthyBjerseyBfilterBdownloadB	dishonestB	comparingBarchitectureBalumniBwearBurbanBupdatesBultimateBscoreB
questionedB	processesBopenlyBmyspaceBmodelsBmanagerBlionBlegendBindexBcostsB	censoringBbodiesBshorterBrushBromeBprivacyBprickBnamelyBmathematicalBlayoutBintroducingB	increasedBfundamentallyBerasedBemoBeatingBdumbassBconsiderableBvisitedBuntaggedBtherapyBtasteBsyntaxBsriBsomeBratingBoverwhelmingBmountainBincorporatedBhominemBdigBdefiningBcreditsBstressBsarahBsafetyBpresentsB
popularityBoccasionallyBinteractionBintentionallyB	followersBexperiencesB	executiveB
complianceBattributionB	answeringBtimBsitushBsikhBsaintBquestioningBpissedBoverlyBnoseBheavenBgodsBgenresBdistinguishB	discoveryBdiscographyBdialogueB
developingBdarwinBcorruptBworriesBthemeBsyndromeBrawBprogressiveB
portugueseBparentBmotherfuckerBlessonBgwenBformulaBdefaultBcorpsBbostonBangerBabusedBwilliamsBsoughtBsevereB	selectionBrollBordersBofferingB	narrativeBmartialB
indicatingB
indefiniteBhatBfailingB	erroneousBcycleBwiderB	userboxesBscrewBottomanBnicelyBnarrowBmarineBjudgingBidealBgainedBfuckedB
deliberateBdefenceB
britannicaBbehaveB	unfoundedBsurnameBstaysBruiningBrantBrankingBkiddingBjrBfootB	elaborateBdisagreementsBdealsBbeginsBalanBaffairsBundidBtwatBslaveryB
resistanceBimhoBfuelBflawedBfestivalBdoucheBcameraBbeatlesBvisionButterBumB	toleratedBspinBsoonerBsensibleB
pretendingBppBpayingBpacificBofficersBnonethelessB
linguisticB	impressedB	hungarianBhumorBdutyB
difficultyB
defamatoryBcreditedBclanBboughtBarmedB
widespreadBvistaBtowerBthrownBthankyouBsufficientlyBstrategyBssBsaturdayB	prejudiceBportionsBorgBmaintenanceBlameBinsaneBimperialBgatherBflatBcombatBbrazilBbogusBbinBavoidingB	automatedBabortionBvietnamBservingBrelateB	opponentsBmarioBluckyBinconsistentBgunsBfridayB	exclusiveBevidentBdicksBdenyingBbrianBawfulBashamedBweaponBuniversitiesBunfreeBtemporarilyB	summarizeBsueBstylebackgroundcolorBstrengthBstockBretardB	pertinentB
perceptionB
maintainedBindiansBincomeB
frustratedBexperimentsB	etymologyBegoBcommunicationsBclassificationBcarsB	botheringBbombBblankedBappleB	academicsBvonBtamilBstrawBsoulB	satisfiedBronBrankBprobabilityBmixBmagicBjeffBholidayBfolkBendlessB
electronicBdnaBconciseBcommonwealthBcolumbiaBcollaborativeBclubsBarthurBangelesB
affiliatedByepBwasntBtrashBsymbolsBpartialBinstitutionsBgovernmentsBeinsteinBdatingBcrashBcombinationBcircleBchemicalBarrivedBwillBversusBpunishedBpoliticiansBplatformBpinkBoffersBnotifiedBmeaningsBliedBgnuB	discussesBcornerB
containingB
contactingBcitizenshipBbitchesBwearingBwakeB
unblockingBsundayBsharingB	sensitiveB
proceduresBplaneB	listeningBliberalsBjustinBguestBgovernorBdaedalusBcrownBconcludeBawardedBapplyingBaidBunawareBtreatyBpreferencesBokayisBmexicanB	mechanismB	maliciousB	magazinesBmacedoniansB
leadershipBknowledgeableBidioticBheadlineBdialectBdeclareBamazonBthreadsB	seeminglyBrankingsBraisingBqueryBpercentB	movementsBimmatureBguardianBdevilBcreatedtookBclauseB	armeniansB	archivingBalbaniaBvirginiaBstormBpresidentialBphaseBnlBmaxBleBgraspB
expressingBerBeducatedBdistributionBcourtsBcaptainBbugBbreedBbreaksBbegunBwpnpaBundoneBswearBsupertrBscientologyBroyBrereadBquietBpursueBpseudoscienceBoregonBontarioBoccursBmonsterBlaborB	interpretB	franciscoB	etiquetteBethicsBdynamicB	disregardB
corruptionBconnectionsBcoffeeBbillionBzealandBwebpageB
suspiciousBsufferB
nominatingBinfiniteB	historiesBhangBguardBgiantBfreakBepicBdollarsBannaB	amendmentBwpcivilBtaiwanB	residentsBqueriesB	proposalsBneedingBkickBjokesB	deliveredB
classifiedBchainBcatsBbordersBbleachanheroBwilsonBterrorB
techniquesBswedenB
restrictedBquantumB
pronouncedBpartlyBnewerBnaBministryBlouisBlegacyBitsB
increasingBhackBgrowthB
exceptionsBdiscoverBcricketBcorporationBconvertB	considersB	chemistryBcharacteristicsBboredBaffairBwikipediaimagesB
volunteersB	variationBtalkbackBsubsequentlyBstudyingBpresentationBmaintainingBkongBfasterBexpiresBemployedBdebatesBdangerB
complainedBcoincidenceBaidsaidsBslowlyBslavicBskyBpalestiniansBobservationsBnewbieBmisuseBmisunderstoodBmirrorBmanageBliftedBhardcoreBhairyBfairnessBextendB	evidentlyBeatsBdiscriminationBdefeatBcrewBcountedB
contradictBbotsBabideB	techniqueBslanderBscottishBrussiansBpushedBoutdatedBoperaBoldestB	occasionsBlebanonB	hypocriteBhintBfindingsB	eliminateBdeletesBconfirmsBclarityB	ancestorsB
accessibleBwronglyBwikipediafilesBweeklyB
uninvolvedBtransferB
traditionsBtonsBthinB	sovereignB	socialismBsaBrowBrepresentingBoffenceBnobelBneilnBmedalBmathematicsBlooseBkennedyBivB	genuinelyBfckBfatuorumBfascismBdecadeB	concludedBcoinB
biologicalBbeerBbearingBarmeniaBannualBamusingBwikipediafairBwelshBvisualBslapB	resultingBregretB	precedentBgamingBemployeeBdividedBdifferBcastleBattendBanyhowBalertByoureBwpagfBworriedBuncitedBtrafficBstealBshouldBselfpublishedBrelationshipsBprophetBpeerreviewedBpaintingBmedievalBlabourBimposeBimpliedBhatesB	functionsBdisorderBdiscouragedB	disagreedBbrowserB	broadcastBborrowedBappropriatelyB
yugoslaviaBvisitorsBtypoBtipBsuccessfullyBserbBrollingB
prohibitedBpleasedBpastedBnotifyBnineBfrustratingB	emotionalBdrivingBdoctrineBconveyBconsentBchargedBbusBboringBalbertBwwiiBweatherBtorontoBtheologyBtemperatureB	stupidityBspiteB
researchedBpulledBmessingBlifetimeBinsightB	hollywoodBfoolwhatBexpiredBdisruptB	dismissedB
cheatsheetBcautionB	brilliantBbreachBbitchmattythewhiteBbiologyBanalogyBaccidentallyB	unhelpfulBtwitterBtiesBtendencyBtargetedB
surprisingBrmBresortBnorrisBlovedBinfringementB	implementBhindiBgamalielBedgeBdemonstratesBculturesBattractBabcBwwBwordedBsuppressBsincereBrepostBrelatesBpresumeBphilB	moderatorBmichiganBmessedBmangaBlossesBinvestigateB	geographyBfrustrationBextensivelyBcookieB	convertedBcoatBbishopBapplicationsByoungerB	witnessesBtrustedBtreatingBstylesB	spreadingBslaveB	selectiveBsecularB
politicianBphilosophicalBobtainBnopeBniggersBnerdBhypocriticalBhumanityBfingerBfightersBexcludeBexamineBendorseB
criticizedBburdenBbareB	transportBtalkcontribsBshipsBsettleBsequenceBiraBintentionalBincidentallyBimproperBftBfarmBemperorBduckB
confidenceBbroaderBbaselessBbangB	authenticBattendedBallianceBadequateBactivistBachievementsB	acceptingByahooBwrightBvirtualBtheresBstoodBsingBrogerBrestrictionsBrefusingBrefusesBquranBprofBprincessBpaB	officialsBmonkeyB
journalismBidentifyingBguessingBfighterBbossBzionistBtBstrongerB	solutionsBsloppyBsenateBsanchezBremovesBreaddBjordanBjointBhamasBformallyBfaqB	discreditBczechBcougarBcontradictionB
contentionBconstitutionalB	civiliansBassemblyBangleBadoptionBwishingBshitholeBselectBscoutBsampleBromaniaB	returningBrepublicansBreminderBreliablyB	reflectedB	recordingBreachingBpatientsB
originatedB	musiciansBlaunchedBhookBhipBharmfulBgreyB
geographicB	employeesBdroppingBdoubtsBdecidingBcuttingBbatmanB
basketballBassaultBassassinationBagreeingBadoptBwpundueBwoodBvolBteachersBtankB	principalBohioB	mythologyBmormonBmodsBmeetupBfedB	estimatesBcelticB
casualtiesBcampusBbugsBbattlesB
azerbaijanBattorneyBalienBwithdrawBwikipediawhereBviolentBtracksBtraceBtechBtalkarticlesBswitchBshittyBreignBrangersBpityBpingBmurderedBmpBknightB	hurricaneB	elizabethBdrivenB	divisionsBdeservedBboundBblacksBappearancesByaaaBunsupportedBunderstandableBtextbookBtendsBrespectfullyB
percentageBparallelB	impartialBharvardBfusionB	forbiddenBfallingBdecidesBchessB	catholicsBapproximatelyBamBalliesBaimedByaaaaahBwinterBwikisBussrBtreesB	splittingBreversedBresearchingBportBpopulationsBownersBoceanBmastersBincorporateBinclinedBillustratesBhuntBhongBhockeyBharrassmentBhaahhahahahBgenerationsB	equipmentBencounteredBdisabledBcriticBcredentialsBcaBbassBwolfBvictoriaB
tournamentBsurgeryBseanBsavingBreserveBpairBkurdishBdipshitBconstructedBcheapBbiznitchBbeachBbackingBzoneBwinnerBviceBupcomingBtrickBtanksBstruggleBslavesBplantsB
pertainingBoverBorganisationsBmrsBmobileBliarsBlawsuitBkellyBironicB	insistingBhopesBhahahaBfoulBflaggedBcrowdB
conversionBcongratsBcheeseiBalphaBadministratorprickB	teachingsBsufferedBspiBslightBroughlyBretainBpraiseBpolitelyBomgBoffendBnounBloudBlaunchBkissBhotelBherebyBgrownBfrancisBfortBdefeatedBcivilianBchristopherBwinnersBwhollyBwhileBtalklistBstoppingBstolenBscheduleBsanctionBruinBrideBrapedBpassesB	partiallyBmerryBmeasuresB
inherentlyB	guaranteeBgrewB	generatedBenvironmentalBcrucialBchancesBbirdBaviationBassertedBwisdomBtropicalBtripleBtransferredBsquadBsoccerBshopBseasonsBrosterBresidentBproducerBplzBomegaBlgbtBjournalistsB
incoherentBimposedBexploreB	equationsB
eliminatedBeditionsBdietB
derogatoryBdannyBcollaborateBcirclesBbullBasksBaccentBxboxBwikiprojectsBwikinewsBwealthBreturnsBraulBpianoBpatrolBnetherlandsBminimalB
manchesterBlandsB	initiatedB	hypocrisyBhopBheadingsBgrayB
girlfriendB	fantasticB	extensionBentitiesBearnedB
disturbingBdescendantsBdennisBcountingBconsistsBconcreteBbuddhismB	associateBwinsBunwarrantedBstayingBsilverB	secretaryBsearchedB	satisfiesBrobB	rightwingBrevengeB
reinstatedB
protestantBobjectivityB
monitoringBkmBkimBjulBjasonB	frequencyBdeclarationBcurrencyBconsequenceBcashB	believingBarguablyB	appointedBanBunsureBunjustifiedBsurviveBsubtleBstickingBsortedBshellBraidBmondayBmaximumBinfluentialBincreasinglyBhandfulBhabitBgothicBforestB	extremistBenjoyedBdustBconstructivelyBchocobosBbeingsB
atmosphereBarBadvocacyBadamsB talkB
worthwhileBwitnessBwhichBvisitingBverbalBtabtabB	supporterBstatisticalBspuriousB
settlementBserversBrumorsB	reviewersBrepresentativesBrayBratioBprogrammingBpreserveBpersiansBpackBnationalistsBliteraryBlinuxBlegitBhispanicBegyptianBassyrianBarrestedBunsubstantiatedBtalkwikiprojectBscandalBrolesB	pakistaniBnomBmythsBmooreBmodBhowardBgossipBgermanicBgenesisBfoolsBestimateBessenceBdrunkBdrawingBdpenisBdisgraceBdisappointedBconsultB	commanderBcomfortableB	appealingB
acceptanceB
wikipediasBviewingBtaylorBstalkerBsearchesBscanBpuertoBprotestsB
propertiesB	mountainsBhatedBharassedB	expectingBdoomB
destroyingBcryingBcontradictoryB	concensusBcommitBcollapseB
canvassingBbruceBassignedBalphabetBactressBworshipB
unlicensedBuncleBtrainedBtenseBscaredBremotelyBpriorityBpetBpantsB
outrageousBknobBhungaryBhoBgravityBgraceBgazaBfoolishBexcusesBearliestBdomesticBdefiniteBdamagingBcopyeditingBcopyeditB	cambridgeBburiedBbeltBapproveBangelBwikipediafeaturedBuhBsortingBsinBsimonB
sanctionedBreaddingBnovelsBmutualBmannBliftB
impressiveBimplicationsBhubBhimherBgeographicalBengineerBdriverBdismissBdestructiveBconcurB
communistsBcivilizationB
boundariesB	automaticBasapB
afterwardsBaeBwhitesBunencyclopedicB
underlyingBthesisBsmellBsilentBrhetoricBredirectingBpluralBobjectedBmatureBliteralBlicenceBjuniorBjonathanBindefBhousesB	hilariousBhealthyBgraduateBgospelBenginesBcritiqueB
convincingBconcertBcalendarBbibliographyBbiblicalB
beneficialB	attributeBadultsBwhereverBweighBtheoreticalBspareBsolarBsenatorBratingsBranksBproducesB	presentlyBpoemBpatrickBpassagesBotherBorientationB	offendingBnewlyBmindedBlewisBkeithBjumpingBjumpedB	intuitiveBguiltBgrammaticalBgeniusBfinlandBdonkeyBdevicesB
definitiveB	criticizeBcoordinatesBcontinuouslyBcellsBcasualBatlanticBafcB
accomplishBwikipedianoBufcBtwBtranscludedBtolerateBtiedBtheseBsubBsimilaritiesB	separatedBroughBpreciousBpartnerBoopsBoiBobsessedB	mainspaceBkeenBkansasBhinduismBhesBfactoBestablishmentBdragonBdisasterBdictatorBdefendedBdebatingBcontradictsB	computeriB	arroganceBanniversaryBworldsBvitalBtouchedBsysopBsuppliedBsufficeBspeculativeBsmearBrugbyBnobleBlovingBlectureB	laughableBhunterBhandlingBgraphicBfyromBfewerBdesiredBcuntlizBcreationismBcleverBchannelsBcardsBbreathBbiteBbansBangelaB	alongsideBagentsBabusesBvanityBtheatreB	sufferingB	spiritualB	sexualityBprotocolBpracticallyBoperatedBoccasionB
navigationBmysteryBmoronsBmootBmentallyBmediumBlistingsBlarryBkickedB
influencesB
industrialBhindusBhazelBgapBgaleBfocusingBfeedBexcludedB	europeansBersBdyingB
disruptingBcuteBcodyBchiropracticBboutBbishonenBsignalBshockedBsciencesBreaddedBpngBplsBpilotBoBmetroBmeasuredBiraniansBimplementedBillustrationB	highlightBescapeBelectricB	douchebagB
compatibleBcomedyBchoosingB	championsB	celebrityBbulletBbroBboardsBatheismBannoyedB	animationB	ambiguityB
adequatelyB  BvfdBtrainsBtortureBtapeBstruckB	shitheadiBscrutinyBrifleBrickBreuseB	resolvingBrealisedBrapidBrailB
proportionB
professionBphotographerBperiodsBnormanBmesanBmarkupBlogsBlesserBlegislationBlabelledB	informingB
incredibleBinaccuraciesBhtmBhmmmBfundingBforcingBequationBequalityBdressB	confidentB
clarifyingBchulaB
chronologyBchanBcarolinaBcapturedBarmBamateurBallenBwellsBwatersB
vietnameseBvenueB
unfamiliarBsyrianBstripBseatBobserverBmccainBmacBlabelsBkurtBhusbandBholidaysBhobbyBhighwayBgoodnessBgaryB	formationB
disciplineBccBbosniaBbombingBblindlyB	allegedlyB	abilitiesB	wikibreakBwarrenB	warrantedBsweepB	suspicionBslangBrescueBrapidlyBpiratesBpetitionBnutsB
nonneutralBneckBmillerBmemorialBlovelyBkillerB
indigenousBignoresB	goodfaithB	formattedBfillingB	favouriteBencouragingB	democratsBclientBcarryingBbnpBbeforeBbasementBbaptistBbacklogBavenueBassholesBallahBalcoholBairedB	advocatesBaccomplishedByeaBwitBwalmartBversesBuuBtieBtherebyBterriblyBteethBstayedBsrBsonsBsamuelBreporterBpreviewBpoundBpneisBphuqBpensnsnniensnsnBpennnisBpaintB	obsessionBlawyersBhorsesBhomeworkBhdBhawaiiBharrisBfallacyBfacedBencyclopediasBeducateB	desperateBdesistBcrackBcountiesB
confirmingBcentsBanytimeB	adjectiveBadhereBwonderedBwastedBwalkingB	unprotectBunilaterallyBtongueBsurvivedBsingingBshahBsegmentBsealBscratchBrumorBregulationsBpersistBpalinBoutletsBosBmfdB	mechanicsBlincolnBgeneBfffffBfascinatingB
collectiveBbecuaseBbeautyB	activistsBaboveBwarfareBuuuuuuB	testimonyB
substituteBsubjectspecificBsoldierB	singaporeBscareBratesBprofitB
preventingBofflineB	nominatorBnasaBmisrepresentationBmetropolitanBmatthewBmanginaBkenBispBhumourBhivBgenerateBfoundingB
federationBexposeBenjoysBdryBdialectsBdescriptiveBcolonialBclearedBceBcartoonBbulgariaBawkwardBassessedB	abandonedBvirusBvariantBuniversallyB
submittingBspectrumB	specificsBsighB
screenshotBsaleBrvvBrudenessB	reactionsB	privilegeBpoolBphrasingBpgBpalBomittedB
obligationB	namespaceBmotiveBmanipulationBlibelousBhangingBgrandfatherBfairuseBethicalBdisappearedBcosBbosnianBbiasesBbanksBabstractB	wholesaleBtigerBstrikesBstoresBsnowBsimpsonBshockBscoresBpotterBportraitBpledgeB
physicallyBphotographyB	parameterBkkBjackassBitsuckBinteractionsBgardenBepBearBdiagramBdesignationB
depressionB	cougaryouB
continuousBconsiderablyBcomposedBchairBcampsBbuddhistBannBwpcoiBunverifiableBunderstandsBteahouseB
structuresBstevenBsomedayBsikhsBsectBrobotBriotsBrefusalB	producingBpersecutionB
performingBpdB
parametersBmobBkyivBkbBkarlBjunkBimplicationBillnessBheadedBgraveBggBgeneticsBflashB	estimatedBdotBdoctorsBdemographicsBcrossedBcourageBconservativesBcollectBclownBboatBbluesBafdsBadsBwarriorBverseBvariableBuniformBtendentiousBshotsBshiftBsessionBrvB
researcherBquoBproxiesBnintendoBmosqueBmayorBmanuallyB
liberationBjaneB	inventionBfurtherBfordBfateBenteringBdancingBcurryBcrusadeBburningBbracketsBbeatingBbarackBassessB
apologizedBairlinesBagenciesB	adventureBwitchBvehiclesBtrendBtestedB	suspendedBsuckingBsucceedBsubstantiveBshoppingBshadowBsatanB	sarcasticBretainedB	reinstateB
presidentsBpieB	oversightBnuBnowadaysBnavalBnakedBmouseBlongtermBlongestBlaughingBintegralB
instrumentBinferiorBincompetentBimaginationBhandedBghostBfifthB	expensiveBestateBestablishingBdeeperBcorrespondingBbreastBbearsBarguesBabBwalterBwaltBunspecifiedB
unofficialBtheoremBtasksBswordBsvgBslimBshoveB	sentimentBsemiprotectionB	satelliteBrossBrefuteB
reflectionB	personnelBoperateBjudgesB
irrationalBideologicalBhehBforkBemployerBdemandsBdelBdebutBcoupB
convenientBcompositionBchoicesBccccccBcapitalizedBcanonBbudgetBbegBbadgeBthxBtaxesBsydneyBsteelBstampBsidedBsheBsantaBrespectableBpurpleBpromisedB	preservedBpolesBplateBperceiveB	multiplesB	misguidedBmelBmarketsBlebaneseBknockBkkkkkkBkillingsBjoyBjohnnyBjayjgBindependentlyBhurtsBhomepageBfrequentBexposureBexperimentalBexaminationBevolvedBenormousB
disclaimerBcontinentalBconflictingBcolumnsBcoalBclockBchinkBcalgaryBbulkBbesideBacBvoicesBtwistedB	succeededBrussellBrestorationBpunctuationBpsychologicalBprayBpompousBplainlyBphilosopherBpenBotBopponentBofficesBobserveBnoticeboardincidentsBnotablyBnewtonBmoodBlesbianBlabBknoxB	insuranceBhonestyBfontBfinnishBeliteB	defensiveB
defamationBconventionalBconnectBconcentrateB	componentBcarriesB	assertingBanyBaimsBagfBwelcomedBwarmB
variationsB	usernamesB	unwillingBthruBtalktheBsynopsisBsuitedBstalkBstadiumB
specialistBspacesB
smarttajinB	slightestBrollsBreadableBratedB	publicityBpricesBpostingsBpostersBportugalB	phenomenaBpatternsBpashtunsBolympicsBnixonBmoBlicksBkeyboardBinvestigationsBflawsBexpelledB	disputingB	discourseBdelhiB	construedBcompliesBcoachBbrightB	brazilianBbowBautoB	attitudesB	argentinaB	telephoneBtacticBsuedBstubbornB
slimvirginBsatisfactionBromansBroadsBreservedB	requiringB	relativesB	radiationB
preferableBpoetryBpassionBnorwayBnonsensicalBmountB
moderatorsB
millionsixBmarsBmannersBladenBjonBinstantBinsistsB	gayfrozenBfriggenBfingersBfacingB
elementaryBdevelopmentsBcuisineBcousinBconsciousnessBchallengingBcarbonBazeriBaforementionedBabsentBwineBwikipedianonfreeBwheelBunfairlyBtributeBsymptomsBstretchB	shortenedBsaddamBrevealsBreactBperspectivesBpennsylvaniaBparBonesidedBmalleusBmalesBloadedBlinguisticsBlankaBjudgedBisolatedB
investmentBinsignificantBinjuryBgainsBexceptionalBenglishlanguageBdiverseBdiesB	demandingBcomparisonsB	collectedBclineBcircularBcfdB	caribbeanBcapitalizationBbondBbillyB	billboardBawhileB	afternoonByeBwhoreeatBwatsonBverbatimBtypedBtwentyBtargetsBtagsfairuseBtagsfairBsuddenB	speciallyBsovereigntyB
separatelyBsaudiBrestrictionBrestatementBrandomlyBqualificationB	programmeBphilipB	pedophileBobjectivelyBloverBkashmirBironyBinnerBinflammatoryBiconBherBguidesBglanceBfranklinBfilingBespBedieB	curiosityBcrystalBconversationsBcondescendingBcinemaBchaosBcarterBbrigadeBboohooBblofeldBbentBantivandalismBaintByearoldBwpownBviableBvendettaBunregisteredBtwistBsacredB
respectfulBreadilyBprofessionalsB	preventedBparaBoccuredBnycBnotationBnflBmediatorBkindnessBjazzB	incapableB	illogicalBgayfagB
evaluationBenforcedB	emergencyBelonkaBdunnoBdrvBdollarBdivineBdivideBdeliveryBdamnedBcapsBatheistsB
artificialBamendedB
ambassadorBadolfBwikiaB
underneathBtaBsimplerBseatsBscenesBscenarioBsatB	recreatedBpollsBpeteBpanelBoutlinedBolympicB	massacresB	logicallyBleftwingBintroductoryB
integratedBhammerBgulfBgatheredBfloodB	everytimeBearsBdownB	criminalsBcoursesB	continentBconcedeB
comprehendB
comparableBcluelessBchaptersBbranchesBbitterBbeefBbaileyB	backwardsBaxisBarbitrarilyBanneBaffiliationBabrahamBaaBwmfBuploaderBtraitsBthemesBtalentBsupB
similarityBscoredBryulongB
remarkableBpurchaseB
proponentsBpromptlyBpovertyB	placementBpatBotrsB	neutrallyBmdBlukeB	liverpoolBipaBillinoisBgregBflameBfightsBexecutedBdistributedB	disappearBdifficultiesB
cuntfranksBcontemptBchairmanBcaptionsBbasicsBbabywhatBawbBappreciationB
admittedlyBviewersBvacationBunhappyBundergroundB
unbalancedBtonBthickB	textbooksBtertiaryBstreetsB	societiesBsoapboxBsmokeBskillBshakespeareBsexyBserialB	scheduledBrankedBqualificationsB
professorsBproceedingsBpremiseBpreBpeakB
paraphraseBparanoidBoutstandingBopBnoiseBnetworksBmusicianBmurrayBmisleadBmetaB
manipulateBlandingBinsistedB
inadequateBinactiveBhumbleB	hostilityBhoodBhomesBgraphB	featuringBevolutionaryBentiretyBdrinkingBdodoB	districtsBdesBcookiesBcloudBclientsBchronologicalB	childhoodBcelebritiesBcaptureB	belongingB	astrologyBafterB
advocatingBverbBunprotectedBunnecessarilyBtricksBtrackingB	toleranceBtaleB
systematicBswitchedBstraightforwardBracingBpunjabB
positivelyBpoleBoutingB	obnoxiousBmailingBmachinesBlegsB	jeffersonBinvestigatedB	hezbollahBfundsBformatsB	executionB	enforcingBelectronB
discretionB	directorsBdebtBcertificateB
borderlineBbetaBapBandersonBabroadB	wisconsinBvocalBviciousBvaryBuserenigmamanBusableBtwinkleBtemperBsonicBsmokingBsanskritBrulesetBrewroteBrevolutionaryBrevenueBremoteBremindsBregistrationBqueerB
prominenceBoverlapBnotalkBnelsonBnancyBmunicipalityBmisconceptionB	minnesotaBmeasurementB	ludicrousBjourneyBiraqiB	interwikiB	inflationB
inevitableB	indonesiaBgibsonBgearBfontsizeBexBerasingB	emphasizeBeligibleB	efficientBdrmiesB	disturbedBdisagreeingB
continuityBcomfortBcharityBcbsBbbBbarryBatleastBassuredBappealsBwpelB	wikilinksBwikilinkBvotersBtruthfulBtrialsBtransparentBtennisBsandB
revelationBretractBregistryBrefutedBrealmBracistsBquantityBpropositionB	plausibleBpermissionsenBpeacefulBoutputBonceBofftopicB
occasionalBnoticingBnormBmarxBmalikBlosersBlobbyBlatinusBkingdomsBkikeBjessicaBjealousBintactBincivilBhuntingB	graduatedBfruitBflBfilthyBfancyB
facilitiesBfacesB	distortedBcumBcreatureBconsequentlyBclosureBchipBblessBbirdsBbeliveBaolB— precedingByugoslavBwalkerBtruckB
transitionBtradingB	testamentBswitzerlandBsphereBruinedBremindedB	reductionBrapBprintingBpremierBnearbyBmoderateBmentoredBmentorBlibertarianBinterviewedBinternationallyB
homophobicB
homeopathyB
ethnicallyBencyclopaedicBearnBdrewBdominantBdiggingBdictionariesBdanishBcombineB
colleaguesBclickedBchoosesBcarlBbuddhaBboreBbakerB	awarenessBautismB	assistantBarriveB	algorithmB	admissionBachievementB£BwallsBtriggerBthompsonBsystematicallyBsuitsBsoapBsingersBseattleBsaintsBrisingBreliefBreferingBraisesBquestBphilippinesBpardonBoralBoddsBnatoBmoscowBmomentsBmohammedBlocateB	interfereBinformalBhighlightedBgenusBeagleBdrivesBdrinksBdistinguishedB	differingBdeterminingBdebatedBcyberBcraigBconBcoloursBcollinsBclerkBchillBcharacteristicBbytesBbiosBberlinBalliedBacademiaBaaronByallBvaticanBurgentB
unilateralBtuesdayBtedBstringsB	strategicBstopsBspicsBsingularBscrewedBreliesBrecommendationBrawatBpunishBpreventsB
povpushingB	norwegianB	mentalityBmemoriesBmarcBkerryBkannadaBjuryBjurisdictionBjoshuaB
ironicallyBinsufficientBinchBidentificationBhopedBhappilyBglassBfortunatelyBfirmlyBfirefoxBfascistsBextraordinaryBedittingBeaBdreamsB	dominatedBdisplaysBdisambigB	desirableBderBclipBchecksBcastingBbehavingBaustinBarabiaBantonioB
antisemiteB
allegianceBalbeitBakinBaccelerationByupByoursB	unethicalBtragedyBthursdayBsubstantiateBspadeBseptBscrollBsarcasmBpullingB	prohibitsBportrayBpornographyBpastingB
overlookedBneighborhoodBnaiveBmixingBmileBmankindBloginBlawrenceBlabelingBindepthB	imaginaryBhanBgrosslyBgrahamBfoundationsB	encounterBdougBdallasB	creaturesB
convictionBcontrolsB	constructB	connolleyBclutterBcatchingBburnedBbuddiesBbraveBbrainsBarmiesBannouncementBannounceBadmitsBadmireBwouldntBwikipediastubBwelfareB	voluntaryBunambiguousBtroublesB	trademarkBthaiBtensBtalk BspottedBsinnBsheepBrevokedBpoetBorbitBopinionatedBnjgwBniBmotivationsBmariaBlonelyBlikingBlegBjailB	inabilityBimplementationBilBidfBhypotheticalBhisBheartsBguardsBfeministB	excludingBequalsBdinnerBdefenderBdaddyBcurseBcontrollingBcompiledB	competentB	churchillBcharacterizationBazerbaijaniB
approachesB	anarchismBalasBaivB	admittingBybmB	welcomingBwayneBtemplateinfoboxBteBsteppedB	spellingsBrespectivelyBregimentBreasonedBrainBproviderBpressingBpiggyBphiladelphiaBpennyBparksBnutB	nonprofitBmsgB
montenegroBmiltonBmassesBloversBloadsBkyleBjasenmBinhabitantsBinfactBhomosexualsBhandyBgladlyBgaysBfundedBfleetBenjoyingBelevenB	egyptiansB
diplomaticBdickfuckingBdevoteBdeceasedBcroatsBcreekB	computingBcompoundB
complexityB
compellingBchaseBcampbellBbroadlyBbridgesB'ahahahahahahahahahahahahahahahahahahahaByogaB	wikipeidaBwikipediasockpuppetBwhiningBvolumesB	verifyingBtranslatingBtibetBtallBtailB
successionBsubsectionsBstreamBsoftB
signaturesBshiaBrulersBreviseB
retirementBreducingBrangesBrajputsB
publishersB	practicedBpplB	notoriousBnerveBnableezyBmodificationBmightyBmafiaB	librariesBlastedBjudicialBinherentB
identitiesBhostedBhonourB
harrassingBguildBgenitalBfartedBexpenseBevadeBenyclopediaBenglishspeakingBdylanB	donationsBdonaldB	diversityBdependB	declaringBcustomBcoloradoBcoinsBclosestBclinicalB	cleansingBclarkBcharlieB
challengesBcfBcategorizationBbuttonsBbluntBbigotBautobiographyB
authorshipBalikeBacidBwpnorBwillingnessBwikipediapossiblyBvisitsB	undermineB	uncertainBtrickyBthirtyBsubpagesBspanBsimultaneouslyBsheerBruralBrssBrowspanBrotBringsBresumeB
restaurantBrepairBquebecBquarterBpermitBonlyB	objectingBneedlessBneBnaughtyBnateB
misconductB	melbourneBmallBmalaysiaBlicensesBkillsBjuiceBjargonBirsBinvadedBinquiryB	innocenceBinformationsBiamBhorrorBgifB	franchiseBdurationBduBdramaticBdivorceB
discourageBdiffersBdeterminationBdesignerBdeanBctBcruelBcopB	collapsedB	campaignsBbulliesBboundaryBbelovedBbasesBbaldBbacksBaryanBariseB	apartheidBaceB	userpagesBurduBundoubtedlyBtwelveBtrekBtotalitarianBtfdBtemptedBsympathyBsympatheticBstakeBsoundingBsigBshedBsharesBsemenBrightlyBresubmitB	residenceBraymondB	producersBprepareBpolitcalBpersonalitiesBpersiaB	peninsulaBparticleBnumberedBltdB	instantlyB
insistenceBimmigrationBhousingBgraphicsBfundBfreakingBformerlyBfeminismBexileBdittoBdisrespectfulBdifferentiateB
designatedBdemocratBcureBcrappyB
complimentBcmaBcircuitBchronicBceoB	byzantineBbowlBberryBbatBaustrianBassertsBampleB
wiktionaryBwiiBunsubscribeBunbelievableBtourismBthouBthoseB
synonymousBstealingBstaticBsimpsonsBshouldntBronaldBrexBputinBprestigiousBpileBpeersBoccupyBnonexistentB	nevermindBnapoleonB
misterwikiB
mistakenlyBmidnightBmatrixBmassachusettsB	magnitudeBmadridBlongtimeBlilBleftistBjoseBjetB	insertionBhmBgoddamnB	gibraltarB
fuckingabfBfleshBfallenB	evidencedB
enterpriseB
encouragesBdosB
distortionBdictatorshipB
definatelyB	customersBcookBconsistBconquestBconcentrationB
componentsBcodingBbongwarriorcongratualtionsBbigotryBbelleBaffectsByoooBwebpagesBwarrantsBvileB
unverifiedB
uneducatedBultraBsugarBstumbledBstackB
socialistsBskipBsandyBrocketBrewordedBrewordBrevBrelyingB	pronounceBplanetsBpirateBpatentsBparaphrasingBovercomeBosamaBoathBnazismBmottoBmotorBmotherfuckingBmorrisBmonthlyBmetricBmeetingsBmaliciouslyBlucasBlineageBkittenBjerryBhoustonBholesBhardwareBgeeB
fraudulentB	frameworkBfixesBevaluateBebBdragBdoseBdjB	descendedBddBdarknessBcubaBcorrectnessBclothesB	civilizedBcheBcharterBbuyingB
businessesBbastardsBbalkanBartisticBacquiredB
accidentalBwetBwavesBwatBvestedBvariantsBtyposBthrustBswissBsplcB	southeastBsemiBscientificallyBsameBricoBremedyB
relativityBreferedBrantsBrageBprovokeBproductionsBpotBpadB'muahahahahahahahahahahahahahahahahahahaBmodificationsB	misplacedBmildBlowestBlistasB	lightgreyBladiesBkeysB
initiativeBinfamousB
indirectlyBinconvenienceB
illiterateBhostingBfutileBforbesBfacultyBfacilityBestBendorsementBenableB
directionsBdetroitBdamagedBcongressionalBcompeteB
collectingB	celebrateBcapBboneBblurbBbeastBbackgroundcolorlightgreyBbaBaxeBaffordBacronymBwipedBwikistalkingBwikipediasandboxBvincentBvacuumBusertalkBturkBtrimmedBtrendsBtoiletBtoddBticketBthyBthoBsystemicBspeculatoroneB
soundtrackBsharpBrestrictBrespectsB	rejectionBregularsBrecommendationsBrandyBpromoB	profanityB	prematureBpaymentBpassiveBoweBoswaldBorganizeBopposesB	occurringBnytBneilBmetersBmakerB
legitimacyBjforgetBjeremyBisraelisBinspirationB	injusticeB	increasesB
immigrantsB
identifiesBhurryBhittingBglennBgeneralsBfucksBfathersB
fabricatedBexaminedBebonicsBearlBdogmaBdissentB
displayingBdensityBdemonstratingBconnecticutB	competingBcollegesBclearingB
carrots→BbeeBbclassBawaitingBarrowBapolloBagendasBwikipediamanualBviB
unpleasantBummBtnaBtirelessBtediousBtechnologiesBtalesBtabloidBsunriseBstudiosBsonyBsecureBsamplesB	revealingBresistB
rememberedB	rejectingBreinsertBrebelsBrailroadB	protocolsB	privatelyB
presidencyBpovsBpersistentlyBpbsBobsoleteBnewcomerBmisrepresentBmiamiBmedalsB
mechanicalBlongstandingB	lifestyleBkentuckyBjehovahB	interfaceBinfantBideallyBgeekBfwiwBforeskinBfactoryBexpressionsBexcitedB	engineersBeasiestB
duplicatedBdesertBdeemBcowB
coordinateBcommunicatingBcatholicismBbombsBbethBbabaBanonymouslyB	aftermathB	advertiseB	addictionB— ByepesBxdB	vuvuzelasB	vancouverBunsuccessfulBtubeBtimingBtightBsubsetBsteamBstabB	smackdownB	skepticalB	screamingBrulerBrocksBrobinsonBrenownedBrandBproofsBpilotsBpigsBparliamentaryBpainfulBobjectionableBnoncommercialBnicholasBnegroBmonetaryBmercyB	mediawikiBmaherBjihadBitaliansBissuingBiconsBhillsBhasBgradingBgeorgianBganBgabsaddsBffBexodusBeveBevansBessaysBembeddedBeggsBebayBeasterBdublinB
dishonestyBdesperatelyBdecencyBcsBcounselBcontradictedB	conqueredB
commitmentBcommaBcmBcharacterizedBchamberBcentersBcentBcandyBbureauBboilerplateB
bestfrozenBbceBbackgroundcolorseashellBamandaBalteringBafghanBacceptsBwppointBworkshopBwellsourcedBwaitedBverticalBverdictBupholdBunsalvageablyBuncomfortableBswiftB
suppressedBsupermanB
summarizedBsoloBshamefulBsexuallyBseverelyB	sacrificeBroutesBrogueBrecievedB	realisticBrajaBpursuingBpriestBpredictBparkerBoxygenBoverwhelminglyBofficalBmonarchyBmmaBmessyBmegaBmarryBmagneticBloanBlesB	invisibleBindianaBillusionBillegitimateBhiphopBhailB	gibberishBgdpBflawBfabricationBexplorerBexpectationsBembarrassingBeaseBdoorsBdiseasesBdiscreditedB	developerBdenmarkBcriticizingBcreatorsBcreationistB
correspondB
conjectureB
conductingBcomposerBclassifyBcastesBboxingBbattlegroundBauthoritarianByankeesB	withdrawnBwishedBvectorBthierB	targetingBswastikaB
superpowerBsuchBstriveBsiB
separationBsectorBsannseBroutineBrenaissanceBredlinksBpunchBpromptBprofilesB	probationB	predictedB	precisionBpractitionersBpossibilitiesBpleasantB
pejorativeBpediaBparticipantBoptBopportunitiesBnpaBmisreadBmiBloopBlolooolbootstootsBlogosB
liberalismBitalicsB
irritatingB	installedBgwBgriffinB	governingB	gatheringBgalaxyBformingBflowersBfalconBemployBemotionsBechoBdisproveB	debatableBdawnBcustomerBconvenienceBcoloniesBceremonyBcaringBcableBbulliedBbonesBblanketBbilljBbigotedBbashBassumesB
assignmentB	anarchistBadressB
accountingBwpnotBweddingBwarriorsBvictorBunjustBtuBtoyB	thresholdBthereofBthereinBthanxBteluguBsubmissionsBstanleyB	spongebobBsistersBshockingBrobinBretardsB
recreationBrajputB	prisonersB	preparingBpradeshBpostalBpassportBoperationalBoliverBneatB
mysteriousBmyselfBmurdersBmostBmormonsBmilitantB	migrationBmercuryBmediterraneanBmcBmarinesBmBlynchBlodgeBlegionBlegendsBlecturesBjoshBinfantryB
improperlyBhuBheraldBhabitsBgongBgiftBgenesB	gamergateBfocusesBfmBexitBeternalBeagerBdualBdistortBdisappointingBcutsBcornishBcolonyBchileBcenteredBbootBbeckB	barnstarsB
attractionB
atrocitiesB	assyriansBangelsBamericasBaccusesBaccompaniedBaccessedBwpnBwpanBwikipedianeutralB	whitewashBwhaleBwardBventureBunwantedBtribalBthirdlyBtheftBtensionBsusanBsurnamesB	strongestBstrikingBstoredBstonesB
standaloneBsmilingBslurBsemiticBscreenshotsBsavageBretrieveBrephraseBrbBrachelBptBprovocativeBpromptedBprohibitBpitchBoverseasBoperatorB	necessityBmurphyBmtvBmarxistBlinearBlendBjossiBjeffroBivanBinfrastructureBheightsBhansBglobeBgeometryBgateBflickrBfemalesB	exclusionB	electoralBduplicationBdragonsBdoubtfulBdeliverBcvBcolspanB	certifiedB
capitalismB	believersBbeggingB
attributesB	arguementBanimatedBaliasBairlineBafricansBadvertB☎BxxB	witnessedBwirelessBwireBwipeBwashBvowelBvoltageBvaryingBunimportantBunifiedBtrailBtoxicBtouchingB
thoughtfulBtheeBthatcherBteenB
structuredBsomaliaBslipBsetupBsettingsBretireBrenderedBrebuttalB	rebellionBqueBprostitutionB	prevalentBprayerBpitB	organisedB	obsessiveBmumbaiBmosesBmontanaBmisrepresentedB
mechanismsBmanningB
litigationBlisaBldsBkentBjuvenileBjuicyBjacobBislamistBinteractBibnBhoundingBhiredBhelenBheatedBheadersBharoldB	halloweenBgordonB	gentlemanBfuB	esperanzaBenabledBelvisBelectricityBdiscoveriesB	diligenceB	criticiseB
consultingBconsiderationsBcomprehensionBcircusBcheatingBcastroBbutterBbothersBblownBbabiesBannoyBaltBalphabeticalBaligncenterBaffixB✉BwpleadBworryingB
wikiquetteBuncalledBtwinBtheaterBterryB	switchingB	survivingBstanBspoilersBspendsBsoundedBslavsBsizesBseeksBscriptsBsatisfactoryBrocB
rationalesBramdasiaB	provincesBprominentlyBpregnantBpenaltyBoscarBoperatesBoklahomaBnrhpB	nightmareBnationalitiesBmillsBloneBlistenedBlibtardBleninBlaneBjokingBjeBjakeBinvestigatingBimmaBgrudgeB
functionalBfuckbagsBfalunBfailepicBenhanceB
employmentBeeBeddieB	economistBearhartBdubBdjpgB	departureBdatBcorporationsBcontinuationBcontextsBconsumptionB	combiningBcherryBcharacterizeBchBcarolBcacheBbryanBbrutalBbroadcastingB	boyfriendBbillionsBbenjaminBbelatedB
bangladeshBbaitingB	artilleryBarrangementBarrangedBwingsBveracityBvegasBurlsBuncontroversialBtsBtrinityBtransitBtraditionallyBthumbBsomeonesBsockingBsmoothBslutBsignalsB
scripturesBrottenBriceB	reiterateB
recordingsBpredictionsB
precedenceBperryB
patrollingBoutlinesBomissionB	nicknamesBnairsBmoroccoBmisrepresentingB	miserableBmasonicBmaskBlosesBlimpeddickedBlibertarianismBlasBlaroucheB	kurdistanBklanBjeanBinvitingBinteriorB
industriesBinconsistencyBillustratedBhoneyBhoaxesBheroesBhazaraBhawaiianBfulfillB
forgettingBevasionB	electronsBeditwarBecwBdictateB	deservingB
derivativeBdeltaBdawkinsBcubanBchampionshipsB	canadiansB
calculatedB
bulgariansBbreedingB
attractiveBatomicBarentBadjustBabuBwprmBwpciteBwikipediaverifiabilityBwhineB
vindictiveBveteranBthaBteslaB	successorBstylebackgroundtransparentBstylebackgroundBsoilBsociallyB
slanderousBsidawayB	shankboneBseedBsamoaBrodBratBpossessBpolicesBplanesB	partisansBoaklandB	notifyingBnicosiaBnbaBnailBmopBmarieBmainlandB	louisianaB
lithuanianBkiteBkhoikhoiBitbitchBhybridBhungryBhatnoteBgrainBgovtBgoodsBgainingBfrownedBfrBfoodsBfollowupBfirmsBfamiliarizeBexposingBdutiesBdumpBdreamguyB
disrespectBdiscretionaryBdetectBdemographicB
delusionalBdelayedBdbBcustomsBcrossingBcommentatorsBcoinedBcoherentBcholaB	cathedralBbonusBblamingBbiBberletBbeneathBbelongedBbeatenBbatteryBaussieB	attractedBassadBanthropologyBaccordBwikipedianamingBvpBvocalsBvirtueBvikingBunusedBundeleteBtrapBtidyBthrewBsupremacistBsuppressionB
structuralBsidebarBsharonBsharmilaBsettlementsB	scriptureB	routinelyBrewardBretaliationB	republicsBreleventB	releasingBregisteringB
reflectingBreducesBrebelBpromisesB
processingBprevailB
practicingBpoundsB	populatedBpipeB
persecutedBpanBoutlookBorganicBoffwikiBnbcBmothersBmonarchBmisterBmeterBmarylandBmarvelBmalcolmBkolkataB	interveneBintelB
inevitablyBhatefulBhamiltonBgpsBgoreBgbBgandhiBfreelicensedBfittingBfansiteBfanaticBexcessBexcerptBentranceBencylopediaBemergedB
efficiencyBdouglasB	disordersBdisingenuousBderekBdeletionistBdasBcropB	compelledBcolumbusBcaredBbelieverBbarrierBbalkansBbackwardBauBassociationsB	apartmentBaliceBalaskaB†B··B talk ByayBwornBwheelsBvaccineB
transcriptBtechnocracyBtalk•BsynonymBstorageBstoleBspeculationsB	southwestBscaryBromanceBriversBridiculouslyBresignBreaganB	qualitiesB	proceededB	primitiveBpissingB	physicistBparadoxB	paintingsBpaceB	operatorsBnewbiesBmonopolyBmixtureBmissouriBmethodologyBmeasurementsBmarkerB	mandatoryBlightsBlearntBleapB
intimidateBimposingBidealsBhariBhackingBgloryBgeologyBfootageBfluidBfkBexaggeratedB	evidencesBemergingB
electricalB	egregiousBdsB
dimensionsBdialogB
developersBdemandedBcypriotsBcurveBcurationBcouldntB
contextualBcontendB
confrontedB	condemnedBcompromisedBcompetitiveB
committingB	chronicleBcartoonsBcareyBbeleiveBbackupBarrangeB	armstrongBarbitratorsBamyBamazedBaliensBwpsynthBwikipediamiscellanyBwadeBvelocityB
travellingBtoysBthereforBtheologicalBterritorialB	survivorsBsunniBsubstitutedB	subjectedBsternBstatueBsnakeBshirtBshieldBsheetBsdBsalvioBriotBrevisitBresignedBresignationB
repetitionB	recurringBproteinBprohibitionBprcB
playgroundB	photoshopBpedanticBparanoiaBorganBnoahBnigeriaB	neglectedBmgBmasonBlousyBlessonsBiphoneBinventBinterferenceBinstructionB	illegallyBiaB
highlightsBhardyB	happinessBhackedBgovBgestureBfunctioningBfreudBfranzBfoundersBforthcomingBfellowsBfaultsB
extremistsBextractBencouragementBdulyBdriversB
descendantBcycloneBcursoryBcruzBcowardlyBcourianoBconstitutedB	consistedB
conscienceBcheatBceasedBcarlosB
capabilityB	cancelledBcameronBbubbleBbreedsBbothBblowingBbillcjBbeingBbankingBbalticB	autoblockBaustriaBattributionsharealikeBassetsBassetBarchaeologicalBancestorB
aggressionBacknowledgingB	achievingByrBwrtBwrestlerBwpbioBwolvesBwhoisBwageBvaguelyBtrailerBthailandB	telegraphB
supportiveBsubcategoryBstewartBspeltBsmashBslantedBsiblingsBsexistBsensesBscoutingBscamBrivalryBripB	resonanceBproneBposedBpocketBpoBpashtunBpalmBpaintedBopensBomitBnerdsB
needlesslyBnairBmutuallyBmisunderstandBmindlessB	messengerBmanufacturingBmanipulatedBmagicalBliquidB	lightningBlayerBkkkBkarmaBjulianBintellectuallyBhastyBfundamentalistBfrozenBfreemasonryBfossilBfloydBfiringBfamineBendorsedBempiresBeliminationBdrnBdowntownBdominateBdiscB	diagnosisBdepictedBdamBcousinsB
competenceB	companionBcommentatorBcoleBclothingBchzzBcentrifugalB	caucasianBcarrierBbrowsingBbrickBbradBbellamyB	battalionBbailBarnoldB	aristotleBappealedB	appallingBamendBahhBadvisoryBaccomplishmentsBweeBvenomB
usefulnessBunprofessionalBtwinsBtuneBtrumpBtreatsBtracedBtmBtiresomeBtibetanBstifleBsovietsBslideBsixthBsinsBshtB
sentimentsB	semanticsBselfpromotionBscoutsBsarfattiBrodeoBrisesB	rewordingBrevisionistBresetB
repetitiveBremovalsBreckonBramBrabbiBpushersB	purportedB	purchasedB
protestingBprosecutionBprofoundB	pregnancyBpredominantlyBpointersBphiB
permittingBpatentlyBpanicB	organismsBoptionalBnightsB
networkingB
negativelyBmonkeysB	metalcoreBmarshallBmarginBmaoBlutherBloyalBlindaBliableBlanganBkillersBkeralaBkennethB	justifiesBinterestinglyBinstrumentsBinitiateB	immigrantBhurtingBhumorousBhellorBhatingBgrindBgriefB	graduallyBgooglingBfractionBfosterBfingBfifaBextinctBexpireB	editathonBdiscrepancyBdiamondB
diacriticsB	dependentB	deliciousBcoordinatorB	cooperateBconservationB	compliantBcategorizedB	cantoneseBcannabisBcalculationBboeingBbishopsBbidBbewareBbasqueB	attendingBarrivalBalisonB
adventuresB ► ByaleBwoundedBworeB	whicheverB	wednesdayBwedgeBvoluntarilyBveganBurantiaBunstableB
unemployedBuncommonBtrimBtransliterationBtrBtitoBtherBstrippedBstrangerBstemBstartersBspinoffBspecialistsBsalvadorBrwB
rhetoricalBrectifyBreachesBqueueBpuzzledBprophecyB
preferablyBpersuadeBpearlBpathsBparodyBortonBnzB	numericalBngBmurdererBmkBmissionsB	ministersBmemeBmasturbationBlibyaBlastingB
justifyingBintenseBimplementingBigorB	hesperianBhadithBhackerBfussB
friendshipBformulationBfoBfiniteBfagsBevangelicalBemotionallyBedwardsBdistantBdirtBcyrusB
criticisedBcorrespondenceBcontradictingBconsumerBcompilationBcliqueBcavalryBbuckBbrochureBbritneyBbottleBbitcoinBathletesB
approachedBanusBanthemBalabamaB♣BwhaBwannabeBwalkedBvermontBvastlyBvalidateBuserpreciousBuprisingBuntillBtylerBtxBtranscriptsBtokyoBtokenBtocBsuperfluousBsubcategoriesB
strugglingB	storylineBstemsBspyBslursBshoutBsavesBrevisingB
refutationBrecklessBpursuitBprovokedBpretextB	portrayedBphillipsBphelpsBorientedBonwikiB	northwestBnonformattedBnahBmuscleBmoronicBmoldovanBmilkBmediateBmatchingBmanufacturersBlengthsBitnBintermediateB	intendingB
indonesianBincompetenceBhpB
hemisphereBharrisonBgrammaticallyBfloatingBfatalBfanboyBfaggotjéskéBeternityBerrB	enlightenB	empiricalBeliminatingBelderBeffectivenessB
downloadedBdishBdiscloseBdignityBdamagesBcurpsB
culturallyBcounterproductiveBcostaBcorrelationBcontributesBconfusesB	complainsBcocaineB
chroniclesBbauderBavatarB	assessingBaroseBapproachingB
algorithmsB	advocatedBadjustedB©BzooBwrestlemaniaBworkerBwikifyBwikBvowelsB
vocabularyBvariesBusenetBupgradeB
unsuitableB
uninformedBtunnelBtrustworthyBtreasureBtransformationBtanB	talkemailBsuppressingBsummarizingBsticksBstatuteB	spidermanBsketchBsecretsBsaharaBrisksBrickkB	renderingBrelaxBredlinkBreallifeB	purposelyBpriestsBpleaBpalmerB
overturnedBouterB
oppressionBobsceneBobligedBmorganB	modifyingBmillBmaturityB
lunchablesBlimitingBlegislativeBkitchenBjenniferBitunesB	integrateBinjuriesBinjuredBincompatibleBinappropriatelyBhuggleBhintsB	hierarchyBheresBheheBhearsayBgutsBgovernmentalB	finishingBfaultyBexhibitB
earthquakeB	detailingB
derivationBdeniesBdemonBcreamB	copypasteBcooperativeBcooperBcommandmentsB	coalitionBcirculationBcensorsBcafeBcabinetBbuzzBbrosBbeatsB	barcelonaBavoidsBasiansBarticalBarcticB
arbitratorBaclassBabandonBwpaivBwoodsBwikipediahelpBwaBversaB	variablesB	valigntopBunexplainedBumbrellaBturnerB
translatesBtitanicB
tangentialBsurvivalBsurprisinglyBsuckedBsuburbanBsuburbBstagesBspoilerBspecializedBsicBscrewingBrumoursBrotatingB
repositoryBrepB	reluctantB
reinsertedBreichBramblingBpurgeBpunitiveB
provisionsB	processorBpokemonBpointerBplasticB	physicianB
organizingBnovaBmsmBmonumentBmichelleBmeatspinBmarkingBmanufacturerBlloydBlighterBlestBleoBleaguesBkudosB	kshatriyaB	knowinglyBkateBjuanBjesseBistanbulBisisBintervalBinterimBinstitutionalBhangonBgravitationalBgianoBgenBgarrisonBfukBfortuneBforensicBfondBfeeBexcitingBenwikiBensuringB	downrightB	dominicanBdoingB	dimensionBcssBcrowB
consistingBconjunctionB	compoundsBclayBchipsBcheerBcaveBbustBbulletsBbuffaloBbobbyBblastBbindingBbillsBbetacommandBbarrettBbaitBbackgroundsB
authorizedBaswellB	astronomyBarizonaBalqaedaBalfredBafaikBzuckByieldBwpfringeBwhitewashingB
vegetarianB
undeletionBundBtruthsBtinB	tennesseeB	teenagersBtalibanBsyriacB	surrenderBsuckerB	subscribeBstrokeB
standpointBsquadronBspellsBslantBskullBskepticsBsiegeBshoutingBshoulderBselectivelyB
satellitesBsaneBroundsB	reportersB	remindingBpsychiatricBprotestantsB
prostituteBprobableBplotsB	platformsBphrasedBphoenixBphilosophersBpartnersBorleansB	originateBneonaziB	neologismBmontrealBmoduleBmockingBmissileBmargaretBmanipulatingBmakersBlegalityBlaughedBisoBiowaBinventorBintegrationBherbertBhavBgrantsB
graduationBgoshBgapsBfuneralBflewBfinalsBfilipinoBfifteenB	festivalsBfasB	extendingB	explosionB
expeditionB	existanceBexemptB
eurovisionBeuroBesBecBdroveBdrawsBdrasticB
disclosureBdestinyBdemiseB	delightedBdefunctB	decliningBdeadlyBcrankBconfessBcolonBcolomboBcolinBchamarBchadBbureaucratsB
bureaucratB	breakdownBblessedBbinksternetBarbB
agreementsBafricanamericanBabbreviationBwikpediaBwikipediachangingBwikilawyeringBwichBviiBveteransBtorahBterriBterminalBtallyB	talkpagesBtajiksBtadBswingBsupernaturalBsubstantiatedBsolvingBsleepingBscumbagBscalesBscBsalivaBrobertsB	reversingB	reproduceBreopenBregretfullyBrefugeesB
readershipBreadabilityBreactedBrcBraoBramaB
proclaimedBpresumedBparenthesesBparadeBpalaceBozBoutletBossetiaBoedB	observersBnoviceBnotionsBnormsBnewestBncccBnativesBmessiahBmanagersB	launchingBkarateBjfkB	jehochmanBislamophobiaBirresponsibleBintBindoeuropeanBindividuallyBindieBinchesBinadvertentlyBimportBhypeBhighkingBhealingBharveyBguessedBgravesBgrassBgogoBglaringBgatesBfryB
fraternityBflipBfiresBfarceBfanaticsBespnBentertainingBduhBdocumentingBdistinctionsB
disserviceBdetrimentalBdesiresBdepartmentsBdemonstrationB	deceptiveB	deceptionBdarkerBcornwallBcopperB
convertingBconnotationsB
conceptionBconcentratedB	certaintyB	catherineB
categorizeBcatalanBbrotherhoodBboroughBbloggerBbengalBbasingBbarbaroBawaitBauthenticityBarisesBarenaBareillyBarchaeologyBanglicanBahmedBactivismB♦ByerBwikifiedBwhimB	webmasterBvoidBvoicedB
vernacularB	varietiesBunionsBtransmissionBtragicBtossBtimothyBtheoreticallyBsweepingBsubwayBstressedBstereotypesB	speculateBskewedBshamBscannedBrougeBromaBrmsBrfcuBrevivalBrejectsB
regulationBrefreshBrealmsBravenBrampantB	promisingB
proceedingBpolarBpiBpermitsBorphanB
orangemikeBnsaBnorseBnomenclatureBmontenegrinBmomentumBmiceBmenuBmbBmariahBmarcusBluisBloBleaningBlandedBknifeB	judgmentsB	intensityB
instructedBinstallB
initiationB	inclusiveBimmuneBiircBhrB	hospitalsBheterosexualB
healthcareBheadquartersBharmonyBharmlessB	hampshireBgothBframesBflamesBethiopiaBelsesBelephantB	disgustedB	directingBdetectedBdeptBdashBcroatBcreationistsB	cosmologyBcorbettBconvertsB
completingBcircaBcapitalsBcamerasBbureaucracyB	breakfastB
beforehandBbattlefieldBbarsBastronomicalB
associatesBapplesBairingB
advantagesB	accidentsBabbreviationsBzionismBwwiBwpmosBwikipedianotabilityBwickedBvaluedBunityBunbanBunambiguouslyBtrioBtouristBtenderBteachesBtalmudBsuffersBstormsBsteppingBspBsoupBslickBsierraBshoesBsequelBselfappointedB	scatteredBscandinaviaBsafelyBritualBridingBrevisionismBrespondsB	resortingBrefugeeBredirectionBquietlyBprovisionalB
preservingBpiraBphotographersB
philippineBperuBpermissionsB
pedophiliaBpeacockBoverhaulBoddlyB
occurrenceBntBnotwithstandingBnofollowBnicerBmommyBmitchellBmisinterpretedBmileyBmedBmathematicianBmandateBlooselyBlimitationsBliBlensB	legendaryBlaoBlandmarkB
laboratoryBkitBkaBjournalisticBjavaBimpartialityBhyphenBhungBholmesBhillaryBhassanBharrassBhaloBhahahahaBgreetingBgraffitiBgolfBgangsBflightsBfilmographyBfckingBerikB
dougwellerBdisinformationBdigestBdiaryB	diagnosedBdesignsBdenseBdarnBcuBcrushBconvenientlyB
connectingBconfederateB	colleagueBcohenBchunkBcenaBcautiousBcasualtyBcalculationsBcakeBbristolBblamedBbarrelBbarredBbaconBazerisBarielBandhraBalbertaByankBwqaBwoB
withdrawalB
undisputedBtransportationBtransgenderB	timestampBthroneB	testifiedBtenureBteenageBstarringB
srebrenicaBspitBsnideBslamBsitsBsilesiaBsettlersBsererBselfishBscreamBsaturnBrumourBrubberBrewritesBregimesBrecoveryBrecoverB
recognizesB
recipientsBreceivesBreassessmentBquickerBpuppyB	proponentBpropagandistsBplatoB	patrioticBoffenderBobtuseBnoobBnobilityBninaBnhlBngoBnervesBneoBmjBmitBminingB	milwaukeeBmeesBmcdonaldBmccarthyBkrishnaBjwsBjamieBitselfBinvadingBinterferingBidiocyBhughesBhowdyBhonoredBhonoraryB
homophobiaBhollandBhireBhighlightingBgrabBgoofBglenB	gentlemenBgenerousBfurBfuckyBfrontierBfliesBflamingBfilthB
exhibitionBeduardoBectBdrakeBdraggedBdraftedBdjathinkimacowboyB	dissidentBdiasporaB
despicableBdakotaBcroppedBcreepyB	continualB	consultedBcompetitorsBcommerceBclassicsBclashB
circumventBchrisoBchangB
celebratedBcatalogBcanalBbreedersBbooBblpsBbdBbathBbarbaraB
attendanceBartworkBarchitecturalBandemuB	amazoncomBalgebraBaccountableBzionistsBwoundB	wolfowitzBwellestablishedBwallaceBvillainBuntoB
unresolvedB	unanimousBummmmmmmBtyrannyBtwilightB
tremendousBtouristsBtobaccoBtamilsB	supremacyBsumsB
subheadingBsubdivisionsBstyleborderspacingBstanfordB
speculatedBsmarterBsikhismBshivaBshadowsBservantBselfservingB	sandsteinBsagaBsaferBrotationBroeringBrockyBrioB	righteousBrfarBreproductionBrepeatsBreflistBrearBrealyB	realworldBrapperBrallyBrainbowBradiusBpxmarginBpressedBportlandB	poisoningBpoemsBpatriotBpaleBpaganB	orchestraBobligationsBnbB	multitudeBmsnB	measuringBmathematiciansBmarcoBmadonnaBloadingBliverBlightingBlawsuitsBlaptopBkeeperBirvingBinvadeBintimidationBinterpretingBimaginedBhttpwwwBhistoriographyBhaterBgwernolBgrowsBgroveBgrandmotherBgoatBgdBgameplayBfunkB	frivolousBforbidBfluxBflexibleBfilmingBfartB	fantasiesB
falsehoodsBfairyBembassyBembarrassedBdrummerB	doctrinesBdisgustBdiminishBdemoBdefameBcpBconnotationBcomparativeBcoloredBcjBchestBchavezBbypassBbwilkinsBburyBbuBbryantB	breathingB
birthplaceBbaronBbacklashBaveBavBaustraliansBatlasBarchaicBapeBahmadinejadBadoptingBaccommodateB☼ByadavBwtBwpweightBwpafdBwondersB
wikisourceBwikipediadoBwhitmanBwaldorfBverizonB
undeniableBtriangleB
translatorB	transformBtranceBtowersBtornadoBtoothBthugBthankfulBtfaBtemperaturesB
swatjesterBsuspectsB
surroundedBstrawmanB	squeakboxBspeciousB
specifyingBsneakyBsmiledBslaughteredBsineBsimplestBshepherdBsenatorsBsemanticBschemesBsahibBsackBrtBrolledBreservesBrecourseBraciallyBprovocationBprefersB	portrayalBoriginatingBnotilB	nicaraguaBnamecallingBmongoliaBmongolBmississippiBlusB	lithuaniaB	linguistsBlimbaughBlightlyBleanBknightsBkickingBjustifiableBjoshuazBjdBipodBinheritanceBinexperiencedBindirectBinconsistenciesBinaneBhydrogenBhungerBholdersB	headlinesBhannibalBhalfwayBglossaryBforemostBfledBfishingBfeudBfattyB	falsehoodB
extinctionB	exchangesBergoBeminemBdresdenBdonateBdistractingBdirektorBdildoBdieselBdiagramsBderiveBderbyBdeputyB	depictingBdemBcowboyBconformsBcondemnBcompromisesBcompetitionsBcoastalBcliffBclarificationsBclansB	chocolateBcardinalBcannonBbreachesBbongwarriorBbondsB
blackpearlBbannersBbananaBballotB
autonomousBauthoredBattachBassesBarsenalB	anonymityBagricultureB	affectingBadvancesBzoeByoungestByamlaBwivesBwinstonBwindsorBwikipediamediaBwesselyBwealthyBwalshBvinceB
undertakerBtravelsB	travelingBtoursBtoddstBtkBtideBticketsBtiBteenagerBtaxonomyBsynonymsBswornBsubarticlesB	statisticBspammerBsolomonBslotBslashBshortenBshabazzBsfBsegmentsBseedsBsaskatchewanBsanityBrickyBrichardsBrevoltB
referendumBreconstructionBrapistBrantingBrajBpseudoB
possessionBposeBpooB	petroleumBpennBpauseB
passionateBonusBnordicBnicoleBnavigateBmyanmarB	murderersBminusBmilanBlunchBlatitudeBkaneB
judgementsBjoelBjeffreyBinvestedB
inventionsB	inferenceBincorporatingBincomingBhollowBheedBhalBguidedB	govermentBfooBflowerBfiftyBfeedingB	favorableBfastestBexplanatoryBevadingBerroneouslyBemailingBelectronicsBdwarfBdummyBdivorcedBdestinationBderivesBdeedsBcoolingBcontradictionsBconsoleBcolbertBcoachingBciaoB	charlotteBburialBbureaucraticBbudBbrdBbomberBautoconfirmedBanticsBanalyzeBamonBalexandrovichBairplaneBagnosticBabdulBzeByouveBwrappedBwpconsensusBwormBwillyBwikipediasuspectedBwellwrittenBwebsterB
vulnerableBvisuallyB	violatorsBviewerB	unwelcomeB
unreadableBunqualifiedBunlockB
unintendedBunificationBunicodeBundergraduateB	undeletedBttBtortBtollBtiplerBtechnoBtandemBtackleBsurveysBstuartB
startclassBspotsBsportingBspaB
somalilandB	sociologyB
singlepageBshikokuBshBsenseiB	sectarianBsecBscotsBsatanismBroskamB	rooseveltBrogersBrivalB	retainingBresurrectionBresolutionsBresentB	realizingBraB
preventionBpreparationBpoisonBplBpizzaBpixelsBownBowlBoutcomesBobeyB	numberingBnjBmoldovaBmodestBmirrorsB	marijuanaBmanufacturedBmansonB	manhattanBmamaB	maintainsBlucyBlipsBlevyBlatinoBlakesBlackedBkoreansBkiBjzaBjaBinfringeBinferB	indicatorBincestBimprovesB
ideologiesBhugoBhooksB	honorableBhighqualityBheadsupB
generatingBgaugeBgagaBfriedBfreemanB	fragmentsBfeasibleBexceedBessjayBesotericBenvyB
engagementBelegantBdrivelBdonatedBdoeBdividingB
distortingB	distancesBdisregardedB
dismissingBdianaB	detectiveBdeployedB	depictionBdeletionistsBcostumeBconfinedBconcealBcompensationBcommieBcommandsBcollaboratedBclownsBclassedB	clarifiesBcherokeeBchapBcelebrationB	breachingBbrahminsBbradleyBblaBbigfootBbernardBbeckjordBarmorBannouncementsBallyBaffiliationsB	aestheticBadverseBadjustmentsBaccentsB
youcaltlasByadaBwindsBwikipediarequestingBwarnerBvladimirovichB	victorianBverticalalignBvariedB
validationBvainBunworthyB	undertakeBufoBtskBtripsBtransformedBtracesBtodoBtdBsymbolicB	superstarBsuperiorityBstylefontsizeBspearsBsophisticatedB
soapboxingBsmugB
skepticismBsignifyBshuttleBshortcutBshawnBshallowBsethBsessionsBsealsBrpgB	ridiculedBrichmondBreproducingBreinsertingBradarB	prototypeBproprietaryBprofitsB
proceduralB
prejudicedB
predictionBpotatoBpopsB	pollutionBpolicingB
pmandersonBpenisiBpedroB
pedophilesBpatriotsBparenthesisBparaphrasedBoverrideBolBobituaryBnursingBnotificationsBninjaBnepalB	municipalBmtBmoralityBmisusedBmisunderstandingsB
missionaryBminsBmidstBmemoBmathsBmastcellBmarginalBmaidenBmadnessB
likelihoodB	libertiesBlegitimatelyBleakedBleakBlapB	labellingBkimesBkarenBjwBjungleBjerksBisaacBiranicaBintimidatingBintimateBinsiderBinsanityB	inheritedBinducedBindiesBhttpBhopkinsBgeographicallyBgentleB
footballerBfiguringBfenianBfenceBfanboysBfaithfulB	examiningBeventualBenablesBelelandBdynamicsBdungeonsBdrumBdropsBdriftB
dominationB
dissentingBdevilsBdestinationsBdepictsBdepictBdemonsBdeferB
dedicationBdecreeBcynicalBcrestBcoverupBcoulterBcopsBconverseB	contractsB
continentsBcommendB	clevelandBcageBburmaBbrunoBbrooklynBbrethrenBbreadBbrassBbrandonBboldlyB	bickeringBbelgiumBbelgianBavailB	atrociousBassistedBarkBarabianBappB	apologistBanchorBaflBabbeyByuBxxxBwreckBwppovBwpmusicBwordyB
wolfkeeperBwildlifeBwikipediawikipediaB	wikimaniaB	wikepediaBwholeheartedlyBwherebyBwhalesBwearsBweaknessBvisaBvanishBuruguayBunproductiveB
ukrainiansBthomsonBtermedBtearsB
subcultureBstuffedB
strategiesBstirB	stabilityBspeedsB	spaniardsBsoxBsmilesBslovakBslappedBskilledB	shouldersBshapedBsectsBruthBromanticBrlBreversalBreutersB	resemblesB
reportedlyB
rephrasingBredoB	provisionBprestigeBpresleyBpredatesBpraisedBpollingBpointyBplagueB
pittsburghBphotographicBpersistsB
perpetuateBpatersonBpasBpackageBoverturnBottawaBorderingBonwardsBodB
objectivesB	northeastBneglectBnathanBnaduBmsnbcBmockBmisinterpretationBmindsetBmigratedB
meditationBmandarinBmakeupBluckilyBlizBlennonB	laureatesBkellerBjoeyBjjBjediBjamaicaBjamB	isolationB
instructorBinscriptionBindiscriminateBinconvenientBimplicitBidleBidentifiableBicelandB
humanitiesBhumanitarianBhttpwwwyoutubecomwatchvBhopelessBhooverBguiseBgtaBgradesBgovernedBgloriousBgarageBfreakinBfratB
foreignersBfollowerBfistB
fallaciousBfailuresBfactcheckingB	exhibitedBevolveBenthusiasticB	endlesslyBeminentBeatenB	dynastiesB	dreadstarBdominionBdistinctiveBdisregardingBdisambiguateBdeutschBdeniersB
degenerateBdanaBcrushedBcrudeB
creativityB	corruptedBcorrieBcopyviosBconsecutiveB	condensedBcommitedBcolonelBcollectionsBcnBchorusBcholasBchiBchandBchamarsB	carbuncleBcapeBcampaigningBbreachedBbrandingBbrainwashedB	borderingBbmwBbladeBbigotsBbfB
behavioralBbacteriaBautonomyBautoblockedBashleyBanxietyBantichristianBannexedBandroidB	amusementB
ammunitionBagriculturalB
actionableBacornBacknowledgesBaccountabilityByadavsBwsBwormsBwiserBwildlyBwhoaBwaryButBunpublishedBunderwayB
undertakenBulsterBughBuefaB	troublingBtorBtombBtissueBtilBthroatBthanksgivingB
thankfullyBtempBtajikB
suspicionsBsuperficialB	summariseBsuB	stylisticBstrainBstinksBstalkersBspeculatingBsinaB	shorthandBshoeBshiftedBsemiprotectBselfproclaimedBsamesexBroyaltyBriderB
respectingBresemblanceBratsBrapesBralphBrabidBrabbisBpursuedB	publishesB	pseudonymB
provincialB
projectionB	processedBprecludeBprecededBpourBpoppingBpastorB	passengerB	particlesBoverzealousB	orthodoxyBomBnumeralsBnguyenBnewmanBncB	mormonismB
misspelledBmissesBmisconceptionsB	memorableB
meatpuppetBmanifestB	macdonaldBlunaBlopezBlockingBlinguistB
lieutenantBlegislatureBlanceBkirkBkennyBkanjiBjanetBitalicBissrBinlandB
implicitlyBimmenseBhrsBhorizonBhonorsBhomerBhighwaysBguestsBgloballyBgenomeBgeBfruitsBfreaksBflavorB	ferdinandBfeminineBfearsBexcessivelyB
evaluatingBequippedBequateBenablingBemotionBelwoodB	dismissalBdiscoBdisciplinesBderryB
deprecatedBdeafBcruiseB	critiquesBcreepBcowardsBcovertBcouncilsBcopeBcoordinatedBconvictionsBcontactsBconservatismBconservapediaB	consciousBconfrontationBconferencesB	composersBcompileBcollaboratorsBcokeBchickBchelseaBchattingBchameriaBchainsBcaspianBcalculusBcaesarBburtonBburnsBbronzeBboundsBblewBbendBbelowB	behaviorsB	beginnersBbashingBautisticBathleteBathensBassignmentsBarkansasB	architectBaptBambassadorsBalgeriaBaimingBwpnfccBwpgngBwpboldB	wikiquoteBwikipediareliableBwifiBwhoopsBweedBvivaB	venezuelaBveiledBvedicBunneededBukipBtrophyB
treatmentsBtoesBtierB	thumbnailB	threatensBtalentedBswearingB	sustainedBstefanBsteerBstarkBspringsB	sponsoredBsituatedBshadeBsendsBscandalsBsatireBsantosBrouxB	romaniansB
rightfullyBrichieBricanBrelayBramoneBragB
qualifyingBqaedaBpythonBpusherBpunjabiBpsychicB
protestersB
propheciesBpronounB
prevailingBprehistoricBpreachBprankBppsBposingBplugBpillarBperformancesB
perfectionB	parallelsBoverkillBoutsiderBouBorladyBopticsB	offspringB	observingBoakBnuisanceBnounsB
nonenglishBneonazisBnearestB
motorcycleBmonroeBmonitorsBminiBmfBmediocreBmatchedB	massivelyBmardyksBmaliceBmadisonBmacroBlynnBlunaticBlunarB	lowercaseBlotusBlordsB	librarianBlaraBlankanBjpB
javascriptBjamesbwatsonBisleBintersectionBinputsB	immenselyBignBidolBiconicBhugelyBhornBhiyaBhboBhawkB	guitaristBgroupingBgregoryBgravitationBglamB	gillespieBgeeksBgabrielB	forwardedBfisherqueenB	firsthandBfavoredBfacistB	expresslyBexpectationBexamsBestablishesBernstB	emissionsB
economistsBduncanB	dominanceB
distributeBdistinguishingBdiscriminatoryBdisBdevoutBdenverBdenominationsBdeityBdeitiesB	datestampBcovenantBcontraBconfigurationBcompressionBcompassB
colloquialB	cofounderBcloneBclairsentienceB
cincinnatiBchamBcemeteryBcapitalisationBbuuBbrushBbroadwayBbpBboldedBbitingBbinaryBbicycleBbelligerentBbehavedBbarberBazBattributingBateBarchBaramaicBahmadBahaBaggressivelyBadlBacknowledgementB
aboriginalB	abolishedB chat BzeusB	yorkshireByorkerByardByBwpsockBwetherBwestminsterBviiiBvenuesBveinButmostBussBuraniumBuptodateB	unnoticedBunnamedB	unlimitedB	unchangedBturtleBtransformersBthinkersBthingyBthankedBtalkinBsustainBsuroB
stereotypeBspheresBsockpuppetingBslaBskepticBsensitivityBsegaBscreensBsabotageBrosesBroomsBromanizationBrobotsBretailB	repostingBreliedBreeditB
redundancyB	recoveredB	recipientBrebeccaBradialBpufferyBpubBprussiaB	prolongedBprisonerBprefaceBpokerBplasmaBpkkB
physicistsBpaysBpaymentsB
passengersBowningBoutrageBopedBooBnostradamusBniggazBnervousBnephewBmwBmiseryBmiscellaneousBmiguelB	messianicBmayaBmarinoBmanagingBltBlocalsBlobbyingBlbsB	landscapeBkwwBkosovaBkicksBjuliaBjealouslyfavonianBiocBinterveningB	hyperboleBhorriblyBgripBgoddessBgobindBgmBgiftedBgarnerBgarB	frederickBflyerBfecesB
extraneousB	expulsionB	expressesBequilibriumB
enthusiasmB	entertainBendeavorB
emphasizedBedgarBdrawingsBdramaticallyBdraftsBdiskBdisciplinaryB	diplomacyBdemonstrablyBdecimalBcrossesBcozB
converselyBcluesBclipsBchunksBcheekBccbysaBcaveatBcarelessBcancelB	calculateBbusesBbrooksBboxerBbosniaksBboostBbloggersBbatistaBatlantaBathleticBantisemitesBantiamericanBamusedB	allemandeBairportsBacupunctureBaccreditationB	absurdityBwikipageBwhistleBwereBwatchesBviennaBveniceB
vandalismsBvaBupheldBunprovenBuniteBuncertaintyBtroubledBtreasuryBthrashBthiefBthermalB
themselvesB	tenstringBtearBtastesBsurvivorB	submarineBstinkBspinningBspencerBspecificationB	specialtyBspecialcontributionsBsoreBsophieBsneakB
slanderingB	sillinessBsilencedBshawBsellsBsecretlyB
satisfyingBsarekB	runescapeBrigorousBrhythmBretiringBresortedB
registriesBrangerBrabbitBqueensB
psychiatryBproudlyBprogressionB	prochoiceB
prevalenceBpitifulBpicksB	phonologyBphasesBpercentagesBpeelBpatriciaBparishB
paranormalB	paramountBopticalBoohBoffendsB	offendersBofcourseBnutshellBnumeralBneighborBncaaBmultiBmortalBmisspellingBminBmeyerBmeatpuppetsB	marriagesB	malaysianBmajestyBmaineBlulzBlineupBlastsBkumarB	karnatakaBjustineBjpgordonBintifadaBinstallationBinscriptionsBinfectedBindisputableBinclinationBhusseinBhunsBhorridBhoaryB	hispanicsBhesitantBherbBhayesBhappierBhandbookBgridBgraphsBgopBgistBfuckersBframedBfourteenBfortyBforgeryBfluentBflagshipBfitnessBferryBfancruftBextendsBexpiryB	exhaustedBexcludesBescapedBescalateB	entwistleBensignBenlightenmentBenactedBemergeBelitistBeighthBedisonBduxBdrasticallyBdogmaticBdiscountBdigitsB
determinesBdenB	delusionsBdeepakBdealingsBcyrillicBcydeBcrusioBcruftBcrashedB	consumingBcommasBcoldplayB	cognitiveBcocktailBcoachesBcluebotBclearcutBchomskyBchimeBchiefsB	catalogueB
capitalizeB
capitalistB	canonicalBburkeB	bulldozerBbrowseBawfullyBaudacityBassholethisB	aspergersBandreaBanatomyB
anarchistsBanaBagBafBadminstratorB	adherentsB	adherenceBadaptedB
accreditedBzombieBwrathBwmcBwikipediabiographiesB
weaknessesBwashedBwarsawBvedasBupsBunsolicitedBunprotectionBuncyclopediaBtyBtweakingBtroublesomeBtrimmingBtodBthroughB	theoristsBswimmingBsweatBswayBswallowBsungBsudanBstylebackgroundcolorwhiteBstomachBstalkedB
spacecraftBsniperBsloganBsingsB
simplisticBsigmaBshoitBsheriffBsharkBseriousnessB
separatingBselfrighteousBscoringBscansBsatanicBsamoanBrudelyBropeBrippedBretreatBresponsibilitiesB	rereadingB
reproducedBrelistedBrebbeBrapingBramonesBquartersB
quantitiesBpuzzleBprotectsBpreservationBpresentationsBpreliminaryB
prejudicesB	preachingBpragueBpotteryBpornographicBpoppedBpinBpeculiarBparadigmBoldsBnswB
nitpickingBnegotiationsBnasalBnarutoBmongolsBmonarchsB	molestingB	molecularBminimizeBmilitiaB	metallicaBmergesBmelaninB
medicationBloggedinBlipBlickingBleonardBleedsBlaymanBlamaBjzgBjungBjumpsBjoanB
interstateBinstabilityB	inhabitedBinfantsBibmB
hypothesesB
hypocritesBhughBhubbardBhowtoB
horizontalBhistoricityBhelmetB	harrassedBharborBhadntBgrungeB
gratuitousB
governanceBglitchBgeneticallyBgeneralizedBgardensBgaddafiBfullerBfooledBflcBfinanciallyB
farmbroughBeyewitnessesBexplorationBexpandsBexoticB	exercisedBencompassesBedgesBdunnBdrkBdongB	dissolvedBdisruptivelyB	disparageBdislikesB
disabilityB	dictatorsBdeterB	demeaningBdefBdeeBdaftBcyclesBcubeBcouplesBcorrespondsBcornellBcontribBconsiderateB
conclusiveBcompromisingBcollectivelyBcollaboratorBchambersB	chalukyasBcentralizedBcdcBbuttheadBbritBbrandsBbrandedBboydBboilingBbikeBbeijingBbeenBbabylonBbabeBassangeBappropriatenessBanticatholicB
anglosaxonB	analogousB
amendmentsBamdBaltitudeBalterationsB
alterationBalmightyBallegingB	alignmentBalignB	alexikouaB
alcoholismBadrianBadministeredBwpbrdBwoolBwithdrewBwikipediacopyrightBwhereinBwellintentionedB
wellingtonBwagnerBvesselsButahBustBucBtyrantBtruelyBtribunalBtreatiesBtraitBtimelyB
thereafterBthankingBtallestBtacticalBsymphonyB	symbolismB	summarilyBsultanB	strangelyBstandardizedBstampsBspouseB	spokesmanBspeechesBsnowmanBslopeBslippedBsinisterB	signifiedB	shamelessB	senselessBseemingBsasBsarekofvulcanBsanteBsailorBrushedBrulingsBroveBrocketsBrlevseBrianaBreversBrethinkBrestartBresidentialBresembleB
researchesBredactedB
recreatingBrangingBpurposefullyBpseudoscientificBprostitutesBproportionalBprophetsBprolificBprincesBprimalBpretendsBpresumptionBpremBprefixBporBplaystationB	planetaryBphilosophiesBpertainsBperiodicBperceptionsBpairsB	outsidersBottersBoneselfBnwBnonnpovBnodeB	negotiateBmulattoBmonstersB	mongolianBmisinformedBmidwestBmiaBmetaphysicalB
metacriticBmehBmarxismBmalteseBmahBluxuryBlouBloganBllcB	lancasterBkissingBkievBkatrinaBjurisdictionsBjeanneBjackieBinvokeB
introducesB	intellectBhostsBheavyweightBheadacheBhartBguitarsBguidingBgsBgiganticBgardnerBfunctionalityBfreedomsBfpBfinanceB
fellowshipBfeesBfalsifiabilityB	extremismB	extractedBexaggerationB	ethiopianB
escalatingBenhancedBeditorializingBeconomicallyB	doctorateBdistressB	disruptedBdisguiseBdiameterBdenoteB	delegatesBdangersBdameBcoxBcorpseB
contestingBconsultationBconstituentBconfrontationalB
confessionB	comprisedBcompatibilityBcollapsibleBcollageBclumsyBclosesBcitableBcheshireBchalkBcertificationBcaucasusBcategorywikipediansBcasuallyBcarsonBbrowsersBbreastsBbrawlBbracketB	blockableBbitchingBbisexualB
birminghamBbeamBbasinBartcileBarjunBarcBappointmentB	amazinglyBaholeBagkB	aggressorBaffirmativeBadvisorBaddendumByehovahByearbookB
wpreliableBwpmedrsBwpcommonnameBwmBwilBwikipediaarticleBwhosBwheatBwesleyBweightsBweighingBwbBvoterBvitaminB	viriditasButilityBusurpedBusdBunlockedBunderminingBtwitBtwistingBtweakBtpBtouringBthrowsBtherapeuticBtheoristB
terminatedBtapesBtallerBsysopsBswineBsurgeonB
supplementBsummedB
summarizesBsuburbsBsubstitutionB
substancesBsubstB
subarticleBstfuBspreeB
solidarityBsinkingBshowerBshitheadBshilohBrubinBroofBriBrhobiteBrevelationsBresidingBreservationsBrenewedBrememberingBrefactorB
redemptionBrecruitmentBreceiverB
rationallyBqingBpyramidBpronounsB
programmerBpremiereBpowBpokeBplatinumB
physiciansBphilippeBpetaB	persuadedBpaulineBpatchB	overboardBoutedBomittingBoffensesB	obligatedBnoncontroversialB	newcastleBneighboringBnegateB	monitoredBmlBmerchantBmammalsBmaltaBmalkeB
louisvilleBliftingBlibertariansBkhoiBkhameneiBkangarooB	jewelleryBjeromeB
italicizedBinvadersB
inspectionBinciteB
inaccuracyB
imperativeBillustratingBhsBhrafnBhijackedBherringBherdB	heideggerBhavingBhappendBgujaratBguessesB
guaranteedBgrantingBgoogledBglassesB	galleriesBgaelicBfundraisingBfriedmanBpfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomfreedomBfreedBflowsBfirearmsBfavorsBfarmersBfactbookBexhibitsB
estimationBessexBerrantBdvdsBdrainBdisdainBdevelopsB
desysoppedBdentalBdenierB	degradingBdebianBdavidsonB	daughtersB	darwinismBczechoslovakiaBcrawlBcottonBcornBconveysBcongressmanBcongratulateB
conceptualBcomposeB
commandersBcohanimBclosetBcircumstanceBchapelB	celestialBcapabilitiesBcamBcablesBbrockBbreatheBblurayBblockadeBbleedingBblairBbettyBbenedictBbatchBbarringBbaloneyBbahaiBbachelorBassessmentsBarrivingBarchivalBarcayneBantonBantBandrewsBanatoliaBallmusicBaldenBakBafghansBadmiralB±B
wrongfullyBwpverifiabilityB	wpcrystalBwooBwikipedianewBviolinBvhsBvampireBuntitledBunholyBunfitBunemploymentB
undergoingBturretBtraitorBtopologyBtjBtemplesBtemplatenameBtangBtaintedBtabooBsuprisedBstyledBstingBsteroidsBstatureBspiritualityBspiritsBspiderBsmBsingledBshillBshesBshapingBseizedBscrapBscifiBschizophreniaBsangerBsacBrouterBromaniB	robertsonBridersBrepostedB	rephrasedBrecruitBrecognizingB	reblockedBpropagandistBproclaimBpressesBpreposterousB	possessedB	portraitsBpondB
politenessBpointofviewBplethoraBpierreBperpetratedBpermissibleB
peripheralB
performersBpenisesBovertlyBorientalBorgansBorderinchaosBonkaB	notorietyBnothisBnortonBnonconstructiveB	neighborsBnedBnavigationalB
nationallyBmythologicalBmuseumsBmunicipalitiesBmphBmozartB	mortalityBmohammadBmockeryBmisnomerBmildlyBmauriceBmatesBmacauBlouiseBlitBlightenBlgagnonB	lessheardBlayersBlacrosseBkhorasanBkernelBjenkinsBirvineB
irreleventBinteractingBinsureB
innovationBinkB
infringingBinertiaB
indicativeB	incidenceBincarnationBimpressionsBimpactsBilkBhumeBhulkBhoundBhollaBhkBharperBhamletBhamBhaikuBhaaretzBgmtBgigBgermaneB	genseiryuB	generatorB	genealogyBgageBfuckenBfriesBfranksBfoolingBflynnBfluBfearedBfavreBfarmerBfamiliarityB
extractionBexploitBepithetB	enjoyableB	endorsingB
encountersBemilyBembraceBelevatorBdurinBdurhamBdumpedBdudleyBdraftingBdosentBdoomedBdisambiguatedBdickbuttB	detectionB	designersB
dependenceBdenominationBdefendsB	databasesBdankBcrimeaBcotwBconveyedB
completionB	columnistBcolonizationBcolombiaB
collateralBclearsBchinB	chemicalsBbwBbrettBbohraBblowsBblinkBbeyonceBberkeleyBbelfastBbeavisBbarnesB
automobileBattestedBattestB
attachmentBassignBasinineBashleeB
archbishopB
apologisedBanticipationBanonsB
americorpsBamaBalignedBalertsBaffirmBaddictedBacquisitionBaccompanyingB
abundantlyB—  B?zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzB	zimmermanByemenBxxxxBxpBwwfB	wpprimaryBwildhartlivieBwikipediadeletionB
wikilinkedBwhomeverBversedBuploadsBunquestionablyBunixB
unfriendlyB	twentiethBtuningBtrumpsBtreasonB	therapiesBtheologiansBthemselfBtexB
tendenciesBtendedBtechnologicalBtatB
sympathizeBswatBsustainableB
suspensionBsurBsunshineBsubheadingsBsteadyBspottingBspammersBsociologicalBsmoothlyBsizedBsinkB	sinanogluBsimplifyBsilkBshyBshippingBshineBshaneBsandwichB	salvationBrsnBriddledBreworkBrestaurantsBresideB
rediculousBreconstructedBrecieveBrecapBreadingsBrashBrailwaysBquotaB	provokingBprosB
programmesBprofessionallyBprobBprixB
privilegedBpredecessorBporkBplanckBphonyBphantomBpervertB
persuasiveBpersistenceBperpetratorsBparoleBpapalBoutragedBoutburstBoliveBnullBnonstopBnonstandardB
nonmuslimsBnevadaBneighborhoodsBmusterB	monumentsBmodelingBminneapolisBmessesBmesopotamiaBmainpageBmagBmachidaBlocallyBlesbiansBleonBlaxBkneesBkleinBkettleBjulieBjudyBjoséBipccB	inventingBintimidatedB
internallyBinstrumentalBinsigniaBinsBingredientsBinformationalB	infectionBineffectiveBimpulseB	illyriansBillyrianBhudsonB	householdBhimselfBhidesBhatsB
harrasmentBguineaBguevaraBgiantsBgerryB	generatesBgatewayB
fulfillingBfuckheadB
fractionalBflimsyBfilmedBfetalBfelonyBfarmsBfandomB	falsifiedBfactionBexportBexploitationB	exemptionB
excellenceBestonianB	espionageBermB	elevationBeldersBekmanBegosBdurovaBdownhillBdistortionsBdisparagingB	disciplesBdiligentBdevoidBdelusionB	defectiveBdeclaresBdarB	customaryBcultsBctmuB
criticallyBcrispBcraftBcowsB	courteousBcorroboratedBcoriolisB
copyeditedBconsBconradB	completlyBcommissionedB	combativeB	clutteredBclergyBchopinBchasingBceilingBcategorisationBcareersBbutcherBburntBbrBbonnieBbombingsBbombedBboilBblendB	birthdateBbirchB	balancingBbaffledBattractsBaspergerBarrangementsB	argentineBarcadeB
apostropheBanticommunistBansB
annexationB
amateurishBallegeBalexaB	abstractsByellingByahB
wrongdoingBwriteupB	wrestlersBwpnotabilityB	worldviewBwikipediarequestedBwikipediahowBwendyBwattsBwashingBwantaBwalksBvolcanoBvladimirBvisitorB
vegetablesBvcBvanuB
unreleasedBuniBunescoBulteriorBturdBtrusteesBtrpodBtropicsBtrollishBtownshipBtitBtaoBsuperpowersBsuperbBsunlightBsuffixBsubcontinentB	stretchedBstraitBstevensBstefanoBsparksB	spacetimeBsorianoBsomersetBsnippetBsmacksBskippedBsimsBshrineB	sentencedBsentanceBsensorBsemesterBseldomBscyllaB	sculptureBsayinBsaiBrookieBriteBrinpocheBrihannaBridiculeBreworkedB
retractionB	retentionBrestructuringBremixBremediesBreincarnationBreformedBrefernceB	recruitedB
recommendsBrealizationBreactorBreactingBrazorBrajusB
publicallyB
propogandaBpretenseB
presentdayBprepBpremisesBpractiseBpoweredBposesBportraysBpistolBphpBphillipB	pervertedBperthBpedoBpearsonB	partitionBparkingBparadiseB	painfullyBpactBpackedBoutbreakB
ostensiblyBorbitalBoprahBoccupationalB	obtainingB
observableBnomadicBnlpBnicheBnestBndpBnavboxBnashB	mysteriesBmyriadBmozillaBmorrisonB	moonshineBmissilesB
millenniumBmerchandiseBmentBmelodicBmeiBmealBmaskedBmarlowBmanaguaBmalusiaBloyaltyBlitterBliterateBlaosB
lancashireBkuBkitsBkgBkennelBkatieBkarachiBjunctionBizBiswasB	irritatedBirrespectiveB
invalidateB
intolerantB	intervalsBinterruptedBinsertsB
inequalityBindicationsBillustrativeBidsBhullBhowlandBhomelessBhatersBhahahahahahahahahahahahahaBhackneyBgrBgospelsBgorillaBgooseBgooddayBgolanBgiovanniBghostsBgeneralizationBgangingBgamespotBfuzzyBfurryBfruitfulBfrequenciesBfrancoBfloodingBfinaleBfelixBfeBfartherBfactionsBexperiencingBexcerptsBexamBevanBenvironmentsBendureB	encompassBembarrassmentB
eliminatesBeliB	educatingBduplicatingBdressedBdomainsBdiwanBdisproportionateB	disparateB
dismissiveBdiscriminateBdiscreditingBdirectsBdigitBdictatesBdefinitivelyBdeedBdeckB	decidedlyBdaneBdalmatiaBdaleB
cunninghamBcrypticBcriticisingBcrackpotBcoryB
contestantBconfidentialBcollaborativelyBcollaboratingBcoincidentallyBcivilisationBchopBcertificatesBcastsBcarpetB
canterburyB	calendarsBcainBcadBbustedBbusinessmanBbundleBbulletinBbritonsBbrisbaneBbootsBbonBboltonB	bollywoodBboiBboBbleachBblatentBblakeB
biologistsB
biographerBbihBbiffBbeholdBbachB	avoidanceBatomBasholB
articulateBanthropologicalBantagonisticB	annoyanceBanecdoteB	anecdotalBandreasBandreBaluminumBalertedBalarmBagedBadvertisementsB
advertisedBadelaideB	abundanceB−BzoraByiddishBwprfcBwouldayBwittgensteinBwikipBwellerBweakerBwatchlistedBwagesBvulgarB	volcanoesBvigilantBuserfyBupgradedB	unpopularBubuntuBtwainBtvpgBtruceBtriumphB	triggeredBtribuneBtreadBtouchesBtopicalBtoolbarBtomatoesBtitsBthugsBthfBtheoldjacobiteBtemplatedidBtelB	taiwaneseBsuiteBsubscriptionBsterlingBstatisticallyBspikeBsoooBsooBsomoneB
solicitingBsnakesBsmokerBslowerBsloveniaBskyhookBshriBshortestBshopsBshiteBshinyBshiningBshiftingBsherlockBsheeshBseventhBserpentBsensibilitiesB	sebastianBschuminBsammyBsafavidsBropesBronzBrobberyBrevivedB
resentmentB
repressionBrepercussionsBrelianceB
relentlessBrefutingBrefinedBrebuildBrafBquinnBpushesBpuffBpsychologistsBpsychologistBprussianB
protectiveB	princetonBpredatorBportalsBpissesBpianistBpersonaBpentagonBpeiceBparentheticalBpabloBorganizationalB	oppressedBoctaveBoccuringB	occupyingBnyoB
noticeableBnotalkstalkBnomineesBnoelB
nawlinwikiBnamblaBmummyBmpsBmouthsBmotorsBmossadBmoroccanB	misquotedB
miscellanyBmidwayBmickBmewBmerkeyBmaroniteBmanorBlumpBludwigsBloosingBlionsBlinkageB	libellousBlayingBlaurentBlatvianB
largescaleBkuruB
kshatriyasBknockedBkinectBkhannaBkenyaBkantBjudoBjudeaBjudasBjosephusBjetsBirpenB
indictmentBimpoliteBimpersonatingBimbecileBimageryB
hungariansBhondaBhometownBhoaxerBhinderBhidB
helicopterBheeBhankBhaitiBhadBgutB	guidlinesBgrandmasterBgradBgodwinBgoaBglasgowBghettoB
geologicalBgeezBgammaBfrontsBfrogB
formulatedBfootballersBfolkloreBfluffBfibersBfetishBfdaBfavoursBfacialB
explosivesB	explosiveB	evaluatedBethosBestoniaBenslavedBenduringBelectiveBeffectedBdyerBdotsBdomeBdobBdistractionBdistractB
distinctlyBdissertationBdisruptionsB	disguisedBdiscrepanciesBdiscouragingBdisappearanceBdiatribeB	depressedB
demolitionBdecisiveBdeceitB	debunkingBdeadlineBdahnBcustodyB	crusadersBcosmicBcornwellBcorneaBcoolerBcookingB
convolutedB	collisionBcolBclasswikitableBcircuitsB	christinaB
charitableBcentresBcdsBcaltonBcalmlyBburnettBboomBbombersBblondeBblessingB	blacklistBbirthsBbetrayedBbenchBbatsBbarnBaynB	audiencesBattackerBattBassimilationB	assaultedB	ascertainBargumentationB
architectsBanthropologistsBanteBalltimeBaleppoB	agreeableBaerialB	advancingB
adjectivesBadjacentB	actressesBacresBabsorbedBzionBzincBzervasBzenByardsByankeeBwritenBwrBwprsnBwoundsBwmdBwiredBwikihoundingBwikidBwhipBwhBweighedBvizBvergeB	vengeanceB	validatedButilizeBusagesBurgesB	upsettingB	unnotableBuniformsBunconfirmedBunauthorizedB
unansweredBtrumanBtrncBtravisBtransparencyB	transfersBtournamentsBtonesBtidiedBthrillerBtetherBtasBtangibleB	tamperingBtalkjohnB	syntheticBsyllableBswordsBswitchesBsurfacesBsuppliesBstyleverticalalignB
strengthenBsteppeB	stephanieBstandardizationBstaffordBspecsBsobBsnapBslovakiaBsilesianB	sickeningBshredB
shorteningBshoreBshellsBshakeBservantsBserbocroatianBscotBscientologistsBsauceBsalaryBsahrawiBrolandBroflB	retractedBreputedBreprintBreopenedB
remarkablyB	remainderBreidBrefactoringB	redlinkedBredditB
recruitingBrecombinationBrecognizableBrecipeBradiantBquittingB
purchasingBpuppiesBprotagonistBprolifeB	projectedBprejudicialB
precedentsBpostwarBpolemicBpleadingBplankBplainsB
physiologyBphoneticBpetarBpepperB	patientlyBpashtoBpartnershipBparrotB
pakistanisBpainsBourBottomansBomarBohnoitsjamieBnvcBnuffBnonnegotiableBnescioB	nefariousBneatlyBnatBmorrellBmodiBmodesB	milleniumBmetalsBmaxwellBmascotBmarkersBmaritimeB
manuscriptB	magicallyBmaamBlazinessBlaughsBkosherBknowledgableBkhalsaBjusBjohnhistoryBjacketBistBirB	intriguedBintendsBintellectualsBinertialBindulgeBincomprehensibleBimportedBimperialismBifwhenBiberianBhrwBhonoursB	hipocriteB	hindsightBhiatusBheroicBheelBhayBgrumpyBgroupedBgroundedB	greenlandBgreedBgrandeBgpBgoinBgnaaBgestapoB	geocitiesBgazetteBgalacticBgagBfrmB	friedrichBfontsBflandersBfiberBfareBfansitesB
expositionBexploredB
exhaustiveBevolvingBersiBernieBensuingBelfBelectBefficacyBedenBduchessB	dramaticaBdqBdonationBdoiBdocumentariesBdoctoralBdivaBdilemmaBdietyBdickfaceBdevoteesBdestroysBdeludedBdefacingBdeceiveBdarthBdaredBcrusadesBcrossmrBcredenceBcpuB
copernicusB
controllerB	consumersB
consultantBconstantineBconditionalB
concludingBcompilerB
compassionBcommercialsBcommentariesBcloudsB
checkusersBchartedBchapmanBbusparBbunnyBbumpBbuckleyB
brittanicaBboycottBbmkBbloomBbloggingBbloatedBbesantBbelarusBbeardBayersBavailabilityB	aurangzebBauditB	athleticsB	assuranceBassociatingB
assholeyouBaryansBarmourB
arithmeticB
arguementsBarchaeologistsB
antagonistBanglerBangB	ancestralBanarchyBanalyzedBaltetendekrabbeBairwaysBairsBafrocentricB
adolescentBadmiBadiB
acquaintedBaccommodationBabundantB	abortionsBabbasB☏B∇∆∇∆B—  talkstalkB– ByrsByeshuwaByellBwrongsBworkingsBwithdrawingBwipingBwikipediavillageBwikipediadonatingBwigdorBwerBwelcomesBweissBwcwBwaterlooB
watchtowerBwastesBwarcraftB	victoriesBvesselBvenusButilizedBusefullBunintentionalBunconstitutionalBtweaksBtroutedBtraveledBtrashedBtransmittedBtransformerBtobyBtiradeBtenuousBtenthBteensBtarotB
tantamountB	taekwondoBswapBsurgicalBsummitBstreamsB
storylinesBstmBspunBspiralsBsparkedBspammedBsourBsocietalBsnBsmithsonianBslBskylineBskinnedB	silencingBsicknessBshytBshuttingBsharmaBshalomBschismBschiavoBsalientBsailBrumoredBrotsBrootedBreviveBrelaxedBrefutesBreformsBreevesB	recentismBrecalledBrealismBqatarBpuzzlingBpunB	proximityB	protestedB
promotionsB
profoundlyBproclamationB
prioritiesBpretentiousBpreexistingBpredictableBpowderB
portrayingBpoetsBpodBplayableBplatesBplagiarizedBpiotrusBpillBpicassoBphBpertainB	performerBpeasantsB	patrolledBpascalBparleyBpandaBpainterBpackagesBouttaBnukeB
nontrivialBnkB	mzilikaziB	murderingBmultiplyBmountedB
montgomeryBmomsBmoleBmodusB
moderationBmisusingBmiracleBmetaphorBmeccaBmeatpuppetryBmcmahonBmatthewsBmarilynBmarathaB	malayalamBmagnetBmadeupB	macmillanBluBloyalistB	longitudeBlevinBlauraBlaserBkluxBkatyBjkBjayronBismailBisiBirlBinterventionsBintercourseBinterBinsecureBinquisitionBinferiorityB
infallibleBindusB	incumbentBidiomB
hystericalBhysteriaBhumblyBhumanismBhumB
huffingtonB
horrendousBheatingBhavocB	harrasingBhandingBhaltBhahBgymBgurusBgrandmaBgraciousBgoonBgeneticistsBgarciaBgamblingBgallingBfulltimeBfuckwitBforestsBforbidsBfloodsB	flatteredBfiBfarrightB
expresswayBexplainationBexertBevertonB	eubulidesBesteemBerasB
entrenchedBenlighteningB
encryptionBemmyBeddyBecologyBdysfunctionalBdullBdukesB	duckworthBdreadfulBdiscreteBdiplomatBdioceseBdinosaurBdicklyonBdianeBdhudhiBdfB	detrimentB	designingB
depictionsB	denialismBdemonstrationsBdegradeBdalyBdalaiBcypriotB
curriculumBcriesBcrawfordBcounterargumentBcookedBconvolutionB	confessedBcompositionsBcomplimentsB
competitorBcommunicatedBcommissionerBcolouredB
collegiateB	collectorBcodexBcoatsBclusterB	cigaretteBchronologicallyBcholesterolBchlorineBchaoticBcgBcensureBceliaBcattleBcategorizingB	cartridgeBcarrBcaililBcabBbutlerBbullsBbuggingBbucketB	botanicalBbogdanovBblurBblocB	blessingsBbitchmotherBbieberBberbersBbenoitBbaldwinBawwwBawBautoformattingBauthorizationBattainBasleepBashBarseholeBarbsBantsB	antiquityBanglesBangeredBandythegrumpBamuseBamnestyB
alignrightB
accessdateBaccBabuserBabdB✍B♫B™B…B–xenotalkB«talk»B¢BzonesBzodiacByieldsBwwwBwpbandB	willfullyBwilayatB
wikipaediaBwikicommonsBwavingB
watchlistsBwangBvoyageBvogueBvisBvicBvhBverticalaligntopcolorB	utilitiesB
usurpationBurineBunwillingnessB	unusuallyBunlawfulBundesirableBunbannedBtyrolBtroyBtrappedBtransitionalBtrailersBtractBtolerantBtidyingB	thcenturyBthcBtemptingB	televisedBteamingBswiftlyBsuspendBsurplusBsuppositionBsuccinctB
subsaharanB	strengthsB
stepbystepBstatBstaleBsquatBsquaresBsponsorBspoiledBspeediedBsparkBsoulsBsooooBsocratesBsnarkyBsmileyBsmellyBsmallestBsleptBskB	sinhaleseBshaftBsgBsenceBselfimportantBschlaflyB	schedulesBsaraBsapiensBrobustBrfcsBrfasB	rewardingBreverendBreuploadBrestrictiveBresponsiblyB
resonancesBresidesBrepairedBrefereeBreedBrcogBrampageBpubmedB	protectorBprotectionsBprospectiveB
prosecutedBpropBpricksB
preferringBpraisingB	portfolioBpoppersBponyBponderBplayboyBpioneerBpickyBphonesBphobiaBpelosiBparapsychologyBowenB
overweightBoutwardB
otherwordsBorgasmBoffencesBoccultBobarBnvidiaBnudityBnotreBnonissueBnomineeBnkbBmtaBmoralsBmorallyBmontyB	moderndayBmlbBmisrepresentsB
misreadingBmisinterpretingBminesBmartyrBmanufactureBmanuelBmanipulativeBmagnificentBlpgpaBlethalBlenoBlennartBlendsB	lecturingBlampBkvBkupcinetB	knowitallBkcBkatBkarabakhBkappaBkaffBjstorBjpsBjonasBjokerBjohannBjiBjcBjarBisolateBinvestBinvertedB	intuitionBinspireBinsightsBinquireBinnuendoBinmatesBhurtfulBhugsBhogBhiltonB
highschoolBheraldryBheirBheapBhashBhanssenBhannahBhagueBgtB	gregaltonB
graduatingBgracefulBgovernBgoodwillBgerardBgavinBgasesBgamersBfundamentalistsB	fulfilledBfuckunblocklifetimeBfrictionBfraserB	forgivingBforciblyBfleshedBfleeingB
flatteringBflapBfeverBfarmingB	fallaciesBfalklandBfactoidB
expansionsBevBethicBepsilonBeppsteinBentBengagesBemotiveBelderlyBeduB	edinburghB
eastendersB	eachotherBdzierzonBdyksBdykeB
duplicatesBduelBdrumsBdrmBdoinBdodgyBdistributionsB
distractedBdisplacementBdisableB	dinosaursBdifferentialBdggBdevB
detractorsBdescendBdeprivedB
deliveringBdelicateBdefeatsBdefamedBdefactoBdecreaseB
dayewalkerBdaresBdairyBdadrianBcusBcrunchBcrumB
criminallyB	creationsBcrashingBcraneBcrB	cordiallyBcoordinationBconstraintsB	conductorB	concertedBcompactBcombB	collusionBcogentBclimbBclarksonBclairvoyanceB	civilisedBchuckleBchopraBchenBcheaperBchargingBceoilBcasinoBcandidBcambodiaBburstBblueboyBblnguyenBblanksBbipolarBbellsBbektashiBbecasueBbeansB	batteriesBbarneyBavidBautopsyBautobiographicalBastrologicalBasteroidB	arvanitesBarrayBarguableBappeaseB
antiisraelBangloBaliasesBagwB
affiliatesB
aestheticsBaegeanB
adaptationBacronymsBaccomplishmentB	abusivelyBaapByahwehBxyzBxviBxenoBwuBwpspaBwpoutingBwilhelmB	wikileaksBwikicupBwelldocumentedBwardsBvoyerBvirusesB	violentlyBvillainsBunwiseBunwieldyBuntilBunscientificBunprecedentedB
unfinishedB
unexpectedBunequivocallyBunderageBunarmedBtsunamiBtrucksBtrillionBtraumaBtractionBtosaBtopsBtneBtitlingBtimedBthraceBthirteenBtheresaBtheodoreBtextboldBterrainBtemplateuserBtaxationBtatarsBtapBtangentBtallyhoB	talkstalkBtaiBswamigalBsustainabilityB
surrealismBsunnisBsufiBsubgenreB
stretchingB
straightenBstraightarmBstatmentBstarredB	starbucksBspecificationsBspasticBsomthingBsolB	slaughterBskiesBsinebotBshortcomingsBshelfBselfevidentBsbBsangBsainiBruinsBrobbedBroastBritualsBripeBricansBreprimandedBreorganizationBremakeB	reeditingBrecuseBrecollectionBreactionaryB	rajasthanBraidersB	radicallyB
publicizedBprospectB
prosecutorBproposesBpreyBpreventativeBprerogativeBprefixesBprayersB
possessiveBplymouthBplazaBpipesBpilingBpharmaceuticalBpestB	persistedBperpetuatingBpendulumBparkwayB
paedophileBovertBoverlappingB	ouzelwellBoutnumberedBousdB
originatorB
originatesB	operativeBooohB	nurembergBnorthwesternBnopBnonadminBnnB
newsworthyB	negativesBneedleBnarrowlyBnamBmushroomBmunichBmundiBmontageBmonkBmonicaB	moleculesBmlsB	militantsBmhpBmetresBmeinBmcewanBmassageBmasonsBmartyrsBmanuscriptsBmanchuBluaBloansBladinBkoranBkochBkneeBkgbB
kannambadiBjustificationsBjpegBjealousyBinvitesB
intriguingBintoleranceB
interactedB
innovativeBinjectBingBinfinityB
immaterialBimfBillustrationsBiciBibanBhobbiesBhjBheirsBhawkingBhauntedBhatchetBhasteBhasntBgtfoBgrandsonB	governorsBgoatsBgmcBglossBgimpBgeoffBgenealogicalBfuturamaBfreddyBfraternitiesBforwardsBfoolishnessBfloodedBfischerBfiddleBfeebleBfeatBextantBexposesB	exercisesBexecuteB
escalationBequivalenceBenwpBenthusiastsBentertainedB	enjoymentBenergiesB
emphasizesBemperorsBelliottBelaborationBdwellingBdugBdreadedB	dravidianBdraggingBdpBdownplayBdownfallBdixieBdisposalBdisgruntledBdisgracefulBdiscouragesBdisadvantageBdireB
diligentlyBdevotionBdevisedBdeviantBdespiseB	designateBdemiurgeBdembskiBdelistedB	definetlyB	defendersBdedicateB
declineyouBdecayBdaylightBdanielsBdalitBcutandpasteB	customizeB
criticizesBcrescentBcreedBcouchBcorpBcontestantsB
consensualBcongregationBconfrontBconferBcondoneB
compulsiveBcompliedBcompetedBcomparesBcomebackBcoandaBclerksBclassificationsBclarkeB
chancellorBcgraphBcellularBcelebratingBcategorypeopleBcatastropheBcapturesB	canvassedBcanisBcanceledBcaltlasB	byzantiumBbuilderB
broadcastsBbratBbrahminBbooleanBblitzerBblindedB	bilingualBbhBbgBberberBbennyBbengaliBbebackB	barbarianBbaptistsBbanterBbakuBbadfaithBaxBawardingBauntBatmBastonishingB	assigningB	artifactsBaricleBargumentativeBapproximationBappreciatesBapplaudBapostleB
announcingB	analyzingBalfBalertingBakinsBagnesB	actualityBacousticBacidsBacademicallyB	absolutlyB←BzipBzhangBzeldaBzealousBzealotsBzachByawnB
xenophobicBwpiarBwoodyBwoodenBwluBwillieB
williamsonBwikipediacivilityBwiccaBwhatnotBvioB
vigorouslyB
verifiablyBverboseB
vehementlyBvedBvaughanBvargasButteredBuserkhoikhoiBuscBupsideBunjustlyBundertakingB	uncoveredBtuluBtranslatorsBtranscriptionBtranscribedBtradedBtrackedB
townsvilleBtorturedBtornBtolkienBtinaBtickBthesesBtextualB	telescopeBtampaBtalkcontributionsBtakeoverBtabtheBtabsBtabloidsBsweepsB	surprisesB	superheroBstrayBstealthBspuiBspontaneousBspiralB	someplaceBsolitaryBsnipingB	slovenianBslitBslateBskypeB
simulationB	shukumineBshirleyBsexismBsepsisB
selfrevertBsegregationBscjesseyBscienceapologistB	scenariosBsantorumBsadrBsaddenedB
sacramentoBrunnerBrumBrubBrodhullandemuBrobbieBrigidBriggedBrfdBreynoldsBrewardsB	revisitedB
retrospectBretardedyourBresurrectedBrestoresBreproductiveBrentBrenaultBremnantsB
reluctanceBregretsBregainBrefugeB	refrainedBreformationBrccBrandallB	qutbuddinB
queenslandB	qualifierB
quackwatchBqedBpurportsBpurgedBpulseBproteinsBpronunciationsBprohibitingBprogrammersBproductivelyBprintsBprimaBprescriptionBprasadBpmidB
plagiarizeBpistonBpipedBphotonB	perpetualBpenguinB
peerreviewBpedigreeBpeasantBpeBpatsB
patriotismBowingBoutsetBorginalB
ordinarilyBoptedBonscreenBoliBocBnutcaseBnurseBnonintrusiveBnomsBnominalB	nisarkandBninthBnihonjoeBneuralB
neccessaryBncramBnailsBmotionsBmontenegrinsBmoeBmodemBmixesB
mitterrandBmilhistBmichelB	messagingB	mentoringBmelmacBmedeisBmasturbatingBmartinezBmarcosBmappingB	manifestoBmanagesBmalayBmalariaBltteBlooserBlkB	liberatedB	liberallyBleopardBleatherBlbBlaughterBladBkurskBknockingB	kidnappedBkhuzaimaBkershawBkayBjsBjoyceBissBinvokedBinvestigativeB	inventorsBinteractiveBintegersBinstructB
institutedBinitialsBinfosBindistinguishableB
improbableBimagescrotumjpgsuckBimaBhumbugBhugBhoppingBhollyBhocBhippieBhijackBhighlandBheelsBhardestBhangsBhairsBgullibleB
greenhouseB	gratitudeBgrammerB	graduatesBgokuBglantzBgitaBgilbertBgiBgeorgBgearedBgayvnBgaaB	furnitureBfunniestBforgivenessBforgivenB	footnotedBfkingBfiascoBfalsettoB	falklandsBfakedBfaithsBfabulousBexpensesBeverestBeugenicsBethnicitiesBeroticBenwikipediaBentersBenlightenedB	employingBemmaBembracedBemblemBellisBelectromagneticBegotisticalBeddB
economicalB	eccentricBdorothyBdorkBdopamineBdlohcierekimBdiveBditchB	disprovenB	displacedBdiscontinuedB
disappointBdiplomaBdesktopBderexB
defendantsB	deathcoreBdealerB	dbachmannBdaviesBdarlingBdaciansBcwB
cumbersomeBcubicBcrowdsBcroreBcretinB
courthouseBcoughBcorpusBcorkBcontribstalkB
contractedB	conformedB
conflictedBconfederationBconclusivelyB	concludesBconcertsB	commandedBcomedianBcolorfulBcleanerBclausBcitBcircumcisedBchicksBcfsBcdrtoolsBccpBcategorisedB	cataloniaBcapriBcantorB	candidacyBcalvinBbutthurtBbowlingBboldingBblpnBbetrayalBbartBbarriersBbaptizedBbalBbaghdadBazerbaijanisB
autoblocksB	augustineBauctionBattainedB	assembledBarrestsB	approvingBapologizingBapesBantisemeticBannoysBamenB
alexandriaBakbarBahemBafterallB	advisableB	admixtureBadheringBadaptBacquireB	abrahamicB☺B”B boomerBzuByammerBwpbiteBwpaeBwinkBwillfulBwikipediocracyBwikipediaorgBwhitneyBwellmeaningBwearyBwcBviewableB
victoriousBvandalproofB
vandalizesBvalignmiddleBvaccinesBuptightBunsuccessfullyBunrealisticBuniquelyBunionistB	unhealthyBunderstandablyB
underminesB	undergoneBundergoB
unblockenlBunavailableBtylasBtunedBtroopersBtrojanB
transistorBtransBtosserBtlkBtireBthorBtestifyBtenetsBtauntingBsynthB
subspeciesB
subsidiaryBsteinBstarvingBstancesBsquirrelBsquarepantsBspokespersonBspoilBspawnedBsosBsortableBsorelyBsonjaBsolicitBslappingBskiingBsinneedB	sincerityBsilentlyBsignersBshrinkBshivBshiftsBshelterBsheetsBsentryBschmidtBsalamBsafavidBruthlessB
rutherfordBrpBrosaBromeoBromBroBrightfulBridesBricaB	reworkingB	restraintB
restrainedBrestateB
resemblingBrescuedBresBrepresentationsB
reinforcesB
regressionB
refreshingBredneckB
rearrangedBrandiB
prosperityB	prosecuteB	proofreadB
prometheusB
progressedBprofessionsBprofaneBproceedsBprizesB
priesthoodBprestonBpopesB	policemanBpokémonBpokingBpleasBplayoffBperlBperiodicallyB	perinçekBperformsBpegBpeekBpeaksB
peacefullyBpatronisingBpathologicalBorionBoreskesBordinalBorangemarlinBoceansBnudgeBnoteableB	nonnativeBnishkidBnirvanaB
nikumaroroBnikolaBmyersB
mutilationBmutationB	mussoliniBmughalsBmughalBmonikerBmoiBmodulesB
moderatingBmilBmicronationsBmichaelsBmestizoBmercilesslyB
melungeonsBmeltingBmelonBmeganBmcgeddonBmattisseBmathematicallyBmaterBmarthaBmarathonB	maharishiBlungB
longwindedB
locomotiveBlineagesBleslieBleafBlawfulBlarBlangBlanBkumanovoBkneejerkB
kilometersBkhojalyB	katherineBjuliusBjosefBjillBjensenBjeezBjatsBisanBionBinvestigatorsB
invaluableB
instigatedBinsinuationsBinformsB
inferencesBinefficientBindentBinclusionistBimgBideologicallyBibB httprexcurrynetwikipedialieshtmlBhostageB
hopelesslyBhopefulBholdingsBhmsBhighnessBhideousBheadlessBharvestBharmingBharBhannityBhallmarkBhacksBgumB
guitaristsBguinnessBgriffithBgreedyB	goldsteinBgnulinuxBgngBgesturesBgermBgeoBgenevaBgeneralizationsBfucktardBfrancesBforrestB	formulateBflexibilityBfissionBfishyBfinestBfiltersBfilteredBfillsBfaxBfavoringB	fashionedBextraordinarilyBexploitsB	exploitedB
exhaustingB	exceedingB	escalatedBequityBequatingBepsB	entrustedBentailsBendorsementsB	endeavourBelitismBeligibilityB
elaboratedBejBdreamingBdplBdonorBdodoucheBdivinityBdistinguishesBdistillationBdislikedBdishesB	dischargeBdiscernB
disallowedBdioxideBdinBdiconoBdickensB
deploymentBdenialsB	denialistBdelegateB	deficientB
deficiencyBdefianceB
decreasingBdebunkedBdanesBcursedBcrucifixionBcrookBcowboysBcorrespondentBcornersB	conveyingBconstructingBconsolidatedBconsciouslyBconcessionsB	conceivedBconcededB	composingB	compilingBcommandmentBcomicalB
combatantsB	collegialBcolemanBcoincideBcohortsBcloakBclicksBclausesBclBckBcivicBchillumBchemistBcementBcargoBcapitolBcapitaBcantonBcaliberBcairoBbyronBburgerBbuggerBbucksBbruteBbrokawBbrevityBbreadthBbrahmoBbradyBbradfordBbosniakB
bootstootsBbootlegBboliviaBboingBblandBbiharBbernieBbeckhamBbayernB	baltimoreBballoonBavenuesB	auschwitzBattractionsB
attractingB
astrologerBassimilatedBapproxBapprovesBappointB	apostolicB
apologistsB	anthologyB	analogiesB	amsterdamBambroseBaltarBaltaicBaloudBallreadyBalbanyBaffirmationBaeonBadvisorsBadriftB
admirationBadhdBabdullahB♪BàBzeppelinBzealBzappaByangBxiiiBwutBwristBwpspamBwpgaBwpdrvB	workplaceBwikipediacitingBwikinaziB	wikibooksBwhitmoreB
westernersBwelchBweightedBwarredBwankersB	wanderingBvolunteeringBvickersBvettingBvettedBverbsBveilBvanillaB	valentineBuserfiedBunrealBunintentionallyBunidentifiedB	underwoodBummmBuberBtwatsBtshirtsBtripeBtranssexualBtradesBtracyBtoxicityBtossedBtodaysBthunderBthsBthorntonB
temptationBtbaBtaxoboxBtalbotBsurveillanceBsuitabilityB
subversiveBsubtitleBstupidlyBstuntB	streamingBstormieBspreadsBspoilingBspammyBsortaBsockpuppeteerBsocketBsoberBsnowdedBsnitchBsmackBsloanBskopjeB
simplicityBshowcaseBshoudBshortageBshirtsBshenanigansB	sheffieldBshanghaiBsectorsBsealedBscreamsBscrappedBschengenBscandinavianBsamsungBsamplingBsalonBsainisBsailingBsageBrydbergBrusBroyalsBroundedBrivalsBrimBricardoBrewardedBrestingB	resistantBreminiscentBremediedBreinstatingBrefineBreenterBredesignBraveBranjitBquackBpursuantBpurrumBpurdueB	psychoticBpsychoBpspB	proisraelBpremiershipBpredominantBpopularizedBpopperBployBpleasingBplayoffsBplaybackBplaceholderBpimpBpierceBphotographedBperiodicalsBpenalBpeeBpayloadBpatentedB	patagoniaBparticularsBparodiesBpanamaBpacksB	packagingBouchBordainedBoperandiBopennessB	omissionsBolyellerBogB	nutritionB	numberoneBnumBnudeBnprBnoticeboardsBnothereBnooobwhyBnfccBnewsweekB
neighboursBnegotiationB
negligibleBnatalieBnarrowmindedBnaikBmtdnaBmoundBmotorwayBmosquesBmosheBmortonBmistressBmiscarriageBmilburnBmickeyBmetaphysicsBmerriamwebsterBmeredithBmccoyBmasksB
marginallyBmalBmajorsBmaggotBmacedonBlumpedB
loverofartBlockeBlieuBleewayBlearnsBleaningsBlaverBlaurenBlatinosBkittyBkelleyBjsutBjmBjattsBjaredBjamaicanBjakartaBjacquesBirwinB
irrigationB
inuniverseB	intrinsicBinterchangeablyBinflatedBineptB	inductionBindependantBincorporationBinconsequentialB	incentiveBimpressBimpracticalBimpersonatorBimperialistBimmunityBimmoralBikeB	hyperlinkBhuntedBhttpnewsbbccoukBhqB	hopiakutaBhmmmmB	hijackingBhermanB	helpimageBhelicoptersBhassleBhardingBhammeredB
hahahahahaBgustavBgreensBgranthBgrammyBgoetheanBglowingBgloriaBgimmeBgillBgilgalBgiftsB	georgiansB	genocidesB	genitaliaBgcBganderBfuckkBfrescoBfreddieBfoldBflavourBfisherBfinkelsteinB
fictitiousBfelonBfavouredBfauxBfaunaB
fatalitiesBfamouslyBfacsBexplodeBexhaustBexeterBexceedinglyBeugeneB
ethnologueB	endeavorsBencodingBellenB
edjohnstonBeclipseBearningBducksBdrownedBdnbBdmacksBdixonBdiscriminatingBdiscoveringBdiscontinueB	disbeliefB	disbandedBdilB	destroyerBdesmondB
denialistsBdeletionlistBdelawareB	defendantBdefamingB	decreasedBdaringBcurrentsBcurlyB	curiouslyBcryptoBcrueltyBcreatBcountessBcounterpartB	counteredBcortezBcoronerB
copypastedB
copenhagenBconvexBcontributerB	continuumBconsumedBconsumeBconstructionsBconspiraciesBconquerBcongoB	concealedBconanB
comprisingBcommonplaceBcommerciallyBcoatrackBcliffordBclassmessageboxBckatzBcivillyBcivilizationsBcingularB
cigarettesBcidBchosunBchengBchaserBcharacterizingB	chameleonBchalukyaBcentredBcausalBcastlesBcarloBcapitalisedBcallerBcalBbusterBbtBbritsBbreakupBbrakeBboyzBborisBblazonBblacklistedBbeltsBbelittleBbehaviouralBbeginnerBbedroomBbathroomBbasketBbaroqueBbangladeshiB	bandwidthBbahBbagsBbadassB	babelfishBayeB	auxiliaryB	attackersBasylumBasswholeB	assistingBassassinatedBartificiallyBarticulatedBarrivesBarisingBariel♥goldB
archiveorgBapproximateBappalledB
antijewishBanticipatedBanomalyBanalysesBanachronisticBambushBamauryBaltoBalmaBallwaysBaigBadvisingBadopterBadlerBadilBacknowledgmentBabkhaziaBabeBabbreviatedB–  talkByearlyByakBxlargeBwrongfulBwrittingBwpcsdB
wikipolicyBwikiingBwiggerBwhackB	westboundBwebbasedBwattenbergerB	watergateBwarpedBvouchBvodkaBvlachsBveganismBvectorsBvassalBvanishedBusherB
userfutureBurartuBupaB
unorthodoxBundulyBunderstatementB	undefinedBuncooperativeBunblockiBunbelievablyBunaBtudorBtuckerBtsarBtrumpetBtrolledBtremendouslyBtonedBtkdBtidbitBthumbsBthiBtheyreB
theologianBthanBtehranBtbBtarBtaipeiBtactBswanBsuspiciouslyBsupervisionBsupermarketBsullivanB
suggestiveB	succesfulB	strugglesBstrivingBstephanBstarrBstargateBstagedBstadiumsBspillB	sphericalBspecimenBspartucusedB	spaghettiBsolvesBslimyBslavBslackBsimmonsBsicilyBshoutedBshoBshieldsBselenaBsecrecyB	screeningBscreamedBscarceB
scandalousBsayingsBsamariaBsaddleBrustyBrulessimplecomplexBruBribbonBrgbBrevolveBrevokeB	resistingB	repulsionBreprehensibleBrepairsBreorganizedB	remembersBreligiouslyBrehashB	registersBreeditedBreceptorBrebuiltBramirezBrajeshBragingBradicalsBrabbinicBpurportedlyBpurityBpulleyBprudentBprtB
provenanceB	propagateB	promotersBproductivityBprincipalityB	primariesBprepubescentBpremiumBpraisesBpractitionerBpositBpopupBpoloBpolanskiBpolBpodcastBplantedBpipingBpioneersBpicturedBperverseB	pervasiveBperuseBpayneB	palpatineB	overnightBoverlookBoverdueBoutweighBopusBokinawaBoffsetBoffhandBocdBoccurrencesB	obstaclesBoblastBnuancesBnotoriouslyB	notationsBnodB
nineteenthBnigerianBnichalpBniagaraBnebraskaB
nationwideB	mysticismBmuzemikeBmurdochBmultiracialBmsuBmrtBmotsB	motivatesBmotherfuckersBmoresoBmoranBmonksB	monkeymanBmollyBmissionariesBmisrBmiraclesBmelissaB	massacredBmarleyBmarekBmagnumBmadchenBlwBlustBlpB	loyalistsBlolooolBlitteredBlionelBlinnaeusB	liabilityBleighBlcBlaysBlaundryBlansingBlabsBkookBkoBknotsBkirbyBkathyBjupiterBjohnuniqBjanitorBjaffnaBizakB	ironholdsBioBinvokingB	invasionsB
invariablyB	intrusiveB
intervenedBinterchangeableB	injectionBindexedBindecentB	incubatorB	incessantBincaseB
imprisonedBimamBidiosyncraticB
hyphenatedBhyperBhygieneBhydeB
hurricanesBhornyBhooBhondurasBhomicideB	holodomorBhintedBheatherBhandlesB	hammurabiBhackersBgusukuBgusleBguidoB	grievanceBgradualBgovernsBgovernorgeneralBgleeBgithubBghB	geometricBgentlyBgentileBgenghisBgeneraBgasolineBgangstaBgamerBgamecubeBgambiaB	fullertonBframingBframBfpcBfourierBfnBflockBflemishBflagrantBfinishesBfidelBfideBferalB	feministsBfeloniousmonkBfeatherBfatigueBfadiaBfabricationsBexhibitionsB
exercisingB	exemplaryBexcelBeulerBetitis
??
Const_5Const*
_output_shapes	
:?u*
dtype0	*̩
value??B??	?u"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '       '      !'      "'      #'      $'      %'      &'      ''      ('      )'      *'      +'      ,'      -'      .'      /'      0'      1'      2'      3'      4'      5'      6'      7'      8'      9'      :'      ;'      <'      ='      >'      ?'      @'      A'      B'      C'      D'      E'      F'      G'      H'      I'      J'      K'      L'      M'      N'      O'      P'      Q'      R'      S'      T'      U'      V'      W'      X'      Y'      Z'      ['      \'      ]'      ^'      _'      `'      a'      b'      c'      d'      e'      f'      g'      h'      i'      j'      k'      l'      m'      n'      o'      p'      q'      r'      s'      t'      u'      v'      w'      x'      y'      z'      {'      |'      }'      ~'      '      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'       (      (      (      (      (      (      (      (      (      	(      
(      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (       (      !(      "(      #(      $(      %(      &(      '(      ((      )(      *(      +(      ,(      -(      .(      /(      0(      1(      2(      3(      4(      5(      6(      7(      8(      9(      :(      ;(      <(      =(      >(      ?(      @(      A(      B(      C(      D(      E(      F(      G(      H(      I(      J(      K(      L(      M(      N(      O(      P(      Q(      R(      S(      T(      U(      V(      W(      X(      Y(      Z(      [(      \(      ](      ^(      _(      `(      a(      b(      c(      d(      e(      f(      g(      h(      i(      j(      k(      l(      m(      n(      o(      p(      q(      r(      s(      t(      u(      v(      w(      x(      y(      z(      {(      |(      }(      ~(      (      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(       )      )      )      )      )      )      )      )      )      	)      
)      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )       )      !)      ")      #)      $)      %)      &)      ')      ()      ))      *)      +)      ,)      -)      .)      /)      0)      1)      2)      3)      4)      5)      6)      7)      8)      9)      :)      ;)      <)      =)      >)      ?)      @)      A)      B)      C)      D)      E)      F)      G)      H)      I)      J)      K)      L)      M)      N)      O)      P)      Q)      R)      S)      T)      U)      V)      W)      X)      Y)      Z)      [)      \)      ])      ^)      _)      `)      a)      b)      c)      d)      e)      f)      g)      h)      i)      j)      k)      l)      m)      n)      o)      p)      q)      r)      s)      t)      u)      v)      w)      x)      y)      z)      {)      |)      })      ~)      )      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)       *      *      *      *      *      *      *      *      *      	*      
*      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *       *      !*      "*      #*      $*      %*      &*      '*      (*      )*      **      +*      ,*      -*      .*      /*      0*      1*      2*      3*      4*      5*      6*      7*      8*      9*      :*      ;*      <*      =*      >*      ?*      @*      A*      B*      C*      D*      E*      F*      G*      H*      I*      J*      K*      L*      M*      N*      O*      P*      Q*      R*      S*      T*      U*      V*      W*      X*      Y*      Z*      [*      \*      ]*      ^*      _*      `*      a*      b*      c*      d*      e*      f*      g*      h*      i*      j*      k*      l*      m*      n*      o*      p*      q*      r*      s*      t*      u*      v*      w*      x*      y*      z*      {*      |*      }*      ~*      *      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*       +      +      +      +      +      +      +      +      +      	+      
+      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +       +      !+      "+      #+      $+      %+      &+      '+      (+      )+      *+      ++      ,+      -+      .+      /+      0+      1+      2+      3+      4+      5+      6+      7+      8+      9+      :+      ;+      <+      =+      >+      ?+      @+      A+      B+      C+      D+      E+      F+      G+      H+      I+      J+      K+      L+      M+      N+      O+      P+      Q+      R+      S+      T+      U+      V+      W+      X+      Y+      Z+      [+      \+      ]+      ^+      _+      `+      a+      b+      c+      d+      e+      f+      g+      h+      i+      j+      k+      l+      m+      n+      o+      p+      q+      r+      s+      t+      u+      v+      w+      x+      y+      z+      {+      |+      }+      ~+      +      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+       ,      ,      ,      ,      ,      ,      ,      ,      ,      	,      
,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,       ,      !,      ",      #,      $,      %,      &,      ',      (,      ),      *,      +,      ,,      -,      .,      /,      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      :,      ;,      <,      =,      >,      ?,      @,      A,      B,      C,      D,      E,      F,      G,      H,      I,      J,      K,      L,      M,      N,      O,      P,      Q,      R,      S,      T,      U,      V,      W,      X,      Y,      Z,      [,      \,      ],      ^,      _,      `,      a,      b,      c,      d,      e,      f,      g,      h,      i,      j,      k,      l,      m,      n,      o,      p,      q,      r,      s,      t,      u,      v,      w,      x,      y,      z,      {,      |,      },      ~,      ,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,       -      -      -      -      -      -      -      -      -      	-      
-      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -       -      !-      "-      #-      $-      %-      &-      '-      (-      )-      *-      +-      ,-      --      .-      /-      0-      1-      2-      3-      4-      5-      6-      7-      8-      9-      :-      ;-      <-      =-      >-      ?-      @-      A-      B-      C-      D-      E-      F-      G-      H-      I-      J-      K-      L-      M-      N-      O-      P-      Q-      R-      S-      T-      U-      V-      W-      X-      Y-      Z-      [-      \-      ]-      ^-      _-      `-      a-      b-      c-      d-      e-      f-      g-      h-      i-      j-      k-      l-      m-      n-      o-      p-      q-      r-      s-      t-      u-      v-      w-      x-      y-      z-      {-      |-      }-      ~-      -      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-       .      .      .      .      .      .      .      .      .      	.      
.      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .       .      !.      ".      #.      $.      %.      &.      '.      (.      ).      *.      +.      ,.      -.      ..      /.      0.      1.      2.      3.      4.      5.      6.      7.      8.      9.      :.      ;.      <.      =.      >.      ?.      @.      A.      B.      C.      D.      E.      F.      G.      H.      I.      J.      K.      L.      M.      N.      O.      P.      Q.      R.      S.      T.      U.      V.      W.      X.      Y.      Z.      [.      \.      ].      ^.      _.      `.      a.      b.      c.      d.      e.      f.      g.      h.      i.      j.      k.      l.      m.      n.      o.      p.      q.      r.      s.      t.      u.      v.      w.      x.      y.      z.      {.      |.      }.      ~.      .      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.       /      /      /      /      /      /      /      /      /      	/      
/      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /       /      !/      "/      #/      $/      %/      &/      '/      (/      )/      */      +/      ,/      -/      ./      //      0/      1/      2/      3/      4/      5/      6/      7/      8/      9/      :/      ;/      </      =/      >/      ?/      @/      A/      B/      C/      D/      E/      F/      G/      H/      I/      J/      K/      L/      M/      N/      O/      P/      Q/      R/      S/      T/      U/      V/      W/      X/      Y/      Z/      [/      \/      ]/      ^/      _/      `/      a/      b/      c/      d/      e/      f/      g/      h/      i/      j/      k/      l/      m/      n/      o/      p/      q/      r/      s/      t/      u/      v/      w/      x/      y/      z/      {/      |/      }/      ~/      /      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/       0      0      0      0      0      0      0      0      0      	0      
0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0       0      !0      "0      #0      $0      %0      &0      '0      (0      )0      *0      +0      ,0      -0      .0      /0      00      10      20      30      40      50      60      70      80      90      :0      ;0      <0      =0      >0      ?0      @0      A0      B0      C0      D0      E0      F0      G0      H0      I0      J0      K0      L0      M0      N0      O0      P0      Q0      R0      S0      T0      U0      V0      W0      X0      Y0      Z0      [0      \0      ]0      ^0      _0      `0      a0      b0      c0      d0      e0      f0      g0      h0      i0      j0      k0      l0      m0      n0      o0      p0      q0      r0      s0      t0      u0      v0      w0      x0      y0      z0      {0      |0      }0      ~0      0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0       1      1      1      1      1      1      1      1      1      	1      
1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1       1      !1      "1      #1      $1      %1      &1      '1      (1      )1      *1      +1      ,1      -1      .1      /1      01      11      21      31      41      51      61      71      81      91      :1      ;1      <1      =1      >1      ?1      @1      A1      B1      C1      D1      E1      F1      G1      H1      I1      J1      K1      L1      M1      N1      O1      P1      Q1      R1      S1      T1      U1      V1      W1      X1      Y1      Z1      [1      \1      ]1      ^1      _1      `1      a1      b1      c1      d1      e1      f1      g1      h1      i1      j1      k1      l1      m1      n1      o1      p1      q1      r1      s1      t1      u1      v1      w1      x1      y1      z1      {1      |1      }1      ~1      1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1       2      2      2      2      2      2      2      2      2      	2      
2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2       2      !2      "2      #2      $2      %2      &2      '2      (2      )2      *2      +2      ,2      -2      .2      /2      02      12      22      32      42      52      62      72      82      92      :2      ;2      <2      =2      >2      ?2      @2      A2      B2      C2      D2      E2      F2      G2      H2      I2      J2      K2      L2      M2      N2      O2      P2      Q2      R2      S2      T2      U2      V2      W2      X2      Y2      Z2      [2      \2      ]2      ^2      _2      `2      a2      b2      c2      d2      e2      f2      g2      h2      i2      j2      k2      l2      m2      n2      o2      p2      q2      r2      s2      t2      u2      v2      w2      x2      y2      z2      {2      |2      }2      ~2      2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2       3      3      3      3      3      3      3      3      3      	3      
3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3       3      !3      "3      #3      $3      %3      &3      '3      (3      )3      *3      +3      ,3      -3      .3      /3      03      13      23      33      43      53      63      73      83      93      :3      ;3      <3      =3      >3      ?3      @3      A3      B3      C3      D3      E3      F3      G3      H3      I3      J3      K3      L3      M3      N3      O3      P3      Q3      R3      S3      T3      U3      V3      W3      X3      Y3      Z3      [3      \3      ]3      ^3      _3      `3      a3      b3      c3      d3      e3      f3      g3      h3      i3      j3      k3      l3      m3      n3      o3      p3      q3      r3      s3      t3      u3      v3      w3      x3      y3      z3      {3      |3      }3      ~3      3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3       4      4      4      4      4      4      4      4      4      	4      
4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4       4      !4      "4      #4      $4      %4      &4      '4      (4      )4      *4      +4      ,4      -4      .4      /4      04      14      24      34      44      54      64      74      84      94      :4      ;4      <4      =4      >4      ?4      @4      A4      B4      C4      D4      E4      F4      G4      H4      I4      J4      K4      L4      M4      N4      O4      P4      Q4      R4      S4      T4      U4      V4      W4      X4      Y4      Z4      [4      \4      ]4      ^4      _4      `4      a4      b4      c4      d4      e4      f4      g4      h4      i4      j4      k4      l4      m4      n4      o4      p4      q4      r4      s4      t4      u4      v4      w4      x4      y4      z4      {4      |4      }4      ~4      4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4       5      5      5      5      5      5      5      5      5      	5      
5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5       5      !5      "5      #5      $5      %5      &5      '5      (5      )5      *5      +5      ,5      -5      .5      /5      05      15      25      35      45      55      65      75      85      95      :5      ;5      <5      =5      >5      ?5      @5      A5      B5      C5      D5      E5      F5      G5      H5      I5      J5      K5      L5      M5      N5      O5      P5      Q5      R5      S5      T5      U5      V5      W5      X5      Y5      Z5      [5      \5      ]5      ^5      _5      `5      a5      b5      c5      d5      e5      f5      g5      h5      i5      j5      k5      l5      m5      n5      o5      p5      q5      r5      s5      t5      u5      v5      w5      x5      y5      z5      {5      |5      }5      ~5      5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5       6      6      6      6      6      6      6      6      6      	6      
6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6       6      !6      "6      #6      $6      %6      &6      '6      (6      )6      *6      +6      ,6      -6      .6      /6      06      16      26      36      46      56      66      76      86      96      :6      ;6      <6      =6      >6      ?6      @6      A6      B6      C6      D6      E6      F6      G6      H6      I6      J6      K6      L6      M6      N6      O6      P6      Q6      R6      S6      T6      U6      V6      W6      X6      Y6      Z6      [6      \6      ]6      ^6      _6      `6      a6      b6      c6      d6      e6      f6      g6      h6      i6      j6      k6      l6      m6      n6      o6      p6      q6      r6      s6      t6      u6      v6      w6      x6      y6      z6      {6      |6      }6      ~6      6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6       7      7      7      7      7      7      7      7      7      	7      
7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7       7      !7      "7      #7      $7      %7      &7      '7      (7      )7      *7      +7      ,7      -7      .7      /7      07      17      27      37      47      57      67      77      87      97      :7      ;7      <7      =7      >7      ?7      @7      A7      B7      C7      D7      E7      F7      G7      H7      I7      J7      K7      L7      M7      N7      O7      P7      Q7      R7      S7      T7      U7      V7      W7      X7      Y7      Z7      [7      \7      ]7      ^7      _7      `7      a7      b7      c7      d7      e7      f7      g7      h7      i7      j7      k7      l7      m7      n7      o7      p7      q7      r7      s7      t7      u7      v7      w7      x7      y7      z7      {7      |7      }7      ~7      7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7       8      8      8      8      8      8      8      8      8      	8      
8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8       8      !8      "8      #8      $8      %8      &8      '8      (8      )8      *8      +8      ,8      -8      .8      /8      08      18      28      38      48      58      68      78      88      98      :8      ;8      <8      =8      >8      ?8      @8      A8      B8      C8      D8      E8      F8      G8      H8      I8      J8      K8      L8      M8      N8      O8      P8      Q8      R8      S8      T8      U8      V8      W8      X8      Y8      Z8      [8      \8      ]8      ^8      _8      `8      a8      b8      c8      d8      e8      f8      g8      h8      i8      j8      k8      l8      m8      n8      o8      p8      q8      r8      s8      t8      u8      v8      w8      x8      y8      z8      {8      |8      }8      ~8      8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8       9      9      9      9      9      9      9      9      9      	9      
9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9       9      !9      "9      #9      $9      %9      &9      '9      (9      )9      *9      +9      ,9      -9      .9      /9      09      19      29      39      49      59      69      79      89      99      :9      ;9      <9      =9      >9      ?9      @9      A9      B9      C9      D9      E9      F9      G9      H9      I9      J9      K9      L9      M9      N9      O9      P9      Q9      R9      S9      T9      U9      V9      W9      X9      Y9      Z9      [9      \9      ]9      ^9      _9      `9      a9      b9      c9      d9      e9      f9      g9      h9      i9      j9      k9      l9      m9      n9      o9      p9      q9      r9      s9      t9      u9      v9      w9      x9      y9      z9      {9      |9      }9      ~9      9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9       :      :      :      :      :      :      :      :      :      	:      
:      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :       :      !:      ":      #:      $:      %:      &:      ':      (:      ):      *:      +:      ,:      -:      .:      /:      0:      1:      2:      3:      4:      5:      6:      7:      8:      9:      ::      ;:      <:      =:      >:      ?:      @:      A:      B:      C:      D:      E:      F:      G:      H:      I:      J:      K:      L:      M:      N:      O:      P:      Q:      R:      S:      T:      U:      V:      W:      X:      Y:      Z:      [:      \:      ]:      ^:      _:      `:      a:      b:      c:      d:      e:      f:      g:      h:      i:      j:      k:      l:      m:      n:      o:      p:      q:      r:      s:      t:      u:      v:      w:      x:      y:      z:      {:      |:      }:      ~:      :      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_151907
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_151912
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?5
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
;
_lookup_layer
	keras_api
_adapt_function*
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses* 
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemp mq!mr/ms0mtvu vv!vw/vx0vy*
'
1
 2
!3
/4
05*
'
0
 1
!2
/3
04*
* 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Aserving_default* 
7
Blookup_table
Ctoken_counts
D	keras_api*
* 
* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

^0
_1*
* 
* 
* 
R
`_initializer
a_create_resource
b_initialize
c_destroy_resource* 
?
d_create_resource
e_initialize
f_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	gtotal
	hcount
i	variables
j	keras_api*
H
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api*
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

i	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

k0
l1*

n	variables*
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
*serving_default_text_vectorization_2_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall*serving_default_text_vectorization_2_input
hash_tableConstConst_1Const_2embedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_151745
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst_6*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_152021
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/embedding/embeddings/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_152106ֿ
?

?
-__inference_sequential_1_layer_call_fn_151181

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?u 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_150567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
$__inference_signature_wrapper_151745
text_vectorization_2_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?u 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_149895o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_3_layer_call_fn_151828

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_150204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_151819

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_150191

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
-__inference_sequential_1_layer_call_fn_151158

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?u 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_150211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_2_layer_call_fn_151781

inputs
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_150180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_151772

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_151857
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
;
__inference__creator_151844
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name43894*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
з
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_150870
text_vectorization_2_inputY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	#
embedding_150854:	?u  
dense_2_150858:  
dense_2_150860:  
dense_3_150864: 
dense_3_150866:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2p
 text_vectorization_2/StringLowerStringLowertext_vectorization_2_input*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_150854*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_150164?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_150858dense_2_150860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_150180?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150191?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_150864dense_3_150866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_150204w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151129
text_vectorization_2_inputY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	#
embedding_151113:	?u  
dense_2_151117:  
dense_2_151119:  
dense_3_151123: 
dense_3_151125:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2p
 text_vectorization_2/StringLowerStringLowertext_vectorization_2_input*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_151113*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_150164?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_151117dense_2_151119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_150180?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150262?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_151123dense_3_151125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_150204w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
/
__inference__initializer_151867
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_151807

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
-
__inference__destroyer_151872
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?;
?
__inference__traced_save_152021
file_prefix3
/savev2_embedding_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *)
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?u :  : : :: : : : : ::: : : : :	?u :  : : ::	?u :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?u :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?u :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?u :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_150164

inputs	*
embedding_lookup_150158:	?u 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_150158inputs*
Tindices0	**
_class 
loc:@embedding_lookup/150158*+
_output_shapes
:?????????x *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/150158*+
_output_shapes
:?????????x ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????x Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_150211

inputsY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	#
embedding_150165:	?u  
dense_2_150181:  
dense_2_150183:  
dense_3_150205: 
dense_3_150207:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2\
 text_vectorization_2/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_150165*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_150164?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_150181dense_2_150183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_150180?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150191?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_150205dense_3_150207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_150204w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_restore_fn_151899
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
??
?
__inference_adapt_step_149148
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
StaticRegexReplace_2StaticRegexReplaceStaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
StaticRegexReplace_3StaticRegexReplaceStaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
StaticRegexReplace_4StaticRegexReplaceStaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
StaticRegexReplace_5StaticRegexReplaceStaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
StaticRegexReplace_6StaticRegexReplaceStaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
StaticRegexReplace_7StaticRegexReplaceStaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
StaticRegexReplace_8StaticRegexReplaceStaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
StaticRegexReplace_9StaticRegexReplaceStaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
StaticRegexReplace_10StaticRegexReplaceStaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
StaticRegexReplace_11StaticRegexReplaceStaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
StaticRegexReplace_12StaticRegexReplaceStaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
StaticRegexReplace_13StaticRegexReplaceStaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
StaticRegexReplace_14StaticRegexReplaceStaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
StaticRegexReplace_15StaticRegexReplaceStaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_16StaticRegexReplaceStaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_17StaticRegexReplaceStaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_18StaticRegexReplaceStaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_19StaticRegexReplaceStaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_20StaticRegexReplaceStaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_21StaticRegexReplaceStaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_22StaticRegexReplaceStaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_23StaticRegexReplaceStaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_24StaticRegexReplaceStaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_25StaticRegexReplaceStaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_26StaticRegexReplaceStaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_27StaticRegexReplaceStaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_28StaticRegexReplaceStaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_29StaticRegexReplaceStaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_30StaticRegexReplaceStaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_31StaticRegexReplaceStaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_32StaticRegexReplaceStaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_33StaticRegexReplaceStaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_34StaticRegexReplaceStaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_35StaticRegexReplaceStaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_36StaticRegexReplaceStaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_37StaticRegexReplaceStaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_38StaticRegexReplaceStaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_39StaticRegexReplaceStaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_40StaticRegexReplaceStaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_41StaticRegexReplaceStaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_42StaticRegexReplaceStaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_43StaticRegexReplaceStaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_44StaticRegexReplaceStaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_45StaticRegexReplaceStaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_46StaticRegexReplaceStaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_47StaticRegexReplaceStaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_48StaticRegexReplaceStaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_49StaticRegexReplaceStaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_50StaticRegexReplaceStaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_51StaticRegexReplaceStaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_52StaticRegexReplaceStaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_53StaticRegexReplaceStaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_54StaticRegexReplaceStaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_55StaticRegexReplaceStaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_56StaticRegexReplaceStaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_57StaticRegexReplaceStaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_58StaticRegexReplaceStaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_59StaticRegexReplaceStaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_60StaticRegexReplaceStaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_61StaticRegexReplaceStaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_62StaticRegexReplaceStaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_63StaticRegexReplaceStaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_64StaticRegexReplaceStaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_65StaticRegexReplaceStaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_66StaticRegexReplaceStaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_67StaticRegexReplaceStaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_68StaticRegexReplaceStaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_69StaticRegexReplaceStaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_70StaticRegexReplaceStaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_71StaticRegexReplaceStaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_72StaticRegexReplaceStaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_73StaticRegexReplaceStaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_74StaticRegexReplaceStaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_75StaticRegexReplaceStaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_76StaticRegexReplaceStaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_77StaticRegexReplaceStaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_78StaticRegexReplaceStaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_79StaticRegexReplaceStaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_80StaticRegexReplaceStaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_81StaticRegexReplaceStaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_82StaticRegexReplaceStaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_83StaticRegexReplaceStaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_84StaticRegexReplaceStaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_85StaticRegexReplaceStaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_86StaticRegexReplaceStaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_87StaticRegexReplaceStaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_88StaticRegexReplaceStaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_89StaticRegexReplaceStaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_90StaticRegexReplaceStaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_91StaticRegexReplaceStaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_92StaticRegexReplaceStaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_93StaticRegexReplaceStaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_94StaticRegexReplaceStaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_95StaticRegexReplaceStaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_96StaticRegexReplaceStaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_97StaticRegexReplaceStaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_98StaticRegexReplaceStaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_99StaticRegexReplaceStaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_100StaticRegexReplaceStaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_101StaticRegexReplaceStaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_102StaticRegexReplaceStaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_103StaticRegexReplaceStaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_104StaticRegexReplaceStaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_105StaticRegexReplaceStaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_106StaticRegexReplaceStaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_107StaticRegexReplaceStaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_108StaticRegexReplaceStaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_109StaticRegexReplaceStaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_110StaticRegexReplaceStaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_111StaticRegexReplaceStaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_112StaticRegexReplaceStaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_113StaticRegexReplaceStaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_114StaticRegexReplaceStaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_115StaticRegexReplaceStaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_116StaticRegexReplaceStaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_117StaticRegexReplaceStaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_118StaticRegexReplaceStaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_119StaticRegexReplaceStaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_120StaticRegexReplaceStaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_121StaticRegexReplaceStaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_122StaticRegexReplaceStaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_123StaticRegexReplaceStaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_124StaticRegexReplaceStaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_125StaticRegexReplaceStaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_126StaticRegexReplaceStaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_127StaticRegexReplaceStaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_128StaticRegexReplaceStaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_129StaticRegexReplaceStaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_130StaticRegexReplaceStaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_131StaticRegexReplaceStaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_132StaticRegexReplaceStaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_133StaticRegexReplaceStaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_134StaticRegexReplaceStaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_135StaticRegexReplaceStaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_136StaticRegexReplaceStaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_137StaticRegexReplaceStaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_138StaticRegexReplaceStaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_139StaticRegexReplaceStaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_140StaticRegexReplaceStaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_141StaticRegexReplaceStaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_142StaticRegexReplaceStaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_143StaticRegexReplaceStaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_144StaticRegexReplaceStaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_145StaticRegexReplaceStaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_146StaticRegexReplaceStaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_147StaticRegexReplaceStaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_148StaticRegexReplaceStaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_149StaticRegexReplaceStaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_150StaticRegexReplaceStaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_151StaticRegexReplaceStaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_152StaticRegexReplaceStaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_153StaticRegexReplaceStaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_154StaticRegexReplaceStaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_155StaticRegexReplaceStaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_156StaticRegexReplaceStaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_157StaticRegexReplaceStaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_158StaticRegexReplaceStaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_159StaticRegexReplaceStaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_160StaticRegexReplaceStaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_161StaticRegexReplaceStaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_162StaticRegexReplaceStaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_163StaticRegexReplaceStaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_164StaticRegexReplaceStaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_165StaticRegexReplaceStaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_166StaticRegexReplaceStaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_167StaticRegexReplaceStaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_168StaticRegexReplaceStaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_169StaticRegexReplaceStaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_170StaticRegexReplaceStaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_171StaticRegexReplaceStaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_172StaticRegexReplaceStaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_173StaticRegexReplaceStaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_174StaticRegexReplaceStaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_175StaticRegexReplaceStaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_176StaticRegexReplaceStaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_177StaticRegexReplaceStaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_178StaticRegexReplaceStaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_179StaticRegexReplaceStaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_180StaticRegexReplaceStaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_181StaticRegexReplaceStaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_182StaticRegexReplaceStaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_183StaticRegexReplaceStaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_184StaticRegexReplaceStaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_185StaticRegexReplaceStaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_186StaticRegexReplaceStaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_187StaticRegexReplaceStaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_188StaticRegexReplaceStaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_189StaticRegexReplaceStaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_190StaticRegexReplaceStaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_191StaticRegexReplaceStaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_192StaticRegexReplaceStaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
StaticRegexReplace_193StaticRegexReplaceStaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_193:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_151761

inputs	*
embedding_lookup_151755:	?u 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_151755inputs*
Tindices0	**
_class 
loc:@embedding_lookup/151755*+
_output_shapes
:?????????x *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/151755*+
_output_shapes
:?????????x ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????x Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
__inference_save_fn_151891
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
__inference_<lambda>_1519078
4key_value_init43893_lookuptableimportv2_table_handle0
,key_value_init43893_lookuptableimportv2_keys2
.key_value_init43893_lookuptableimportv2_values	
identity??'key_value_init43893/LookupTableImportV2?
'key_value_init43893/LookupTableImportV2LookupTableImportV24key_value_init43893_lookuptableimportv2_table_handle,key_value_init43893_lookuptableimportv2_keys.key_value_init43893_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init43893/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?u:?u2R
'key_value_init43893/LookupTableImportV2'key_value_init43893/LookupTableImportV2:!

_output_shapes	
:?u:!

_output_shapes	
:?u
?

?
-__inference_sequential_1_layer_call_fn_150232
text_vectorization_2_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?u 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_150211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
*__inference_dropout_1_layer_call_fn_151797

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150191`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_150262

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
__inference__creator_151862
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_33593*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
__inference__initializer_1518528
4key_value_init43893_lookuptableimportv2_table_handle0
,key_value_init43893_lookuptableimportv2_keys2
.key_value_init43893_lookuptableimportv2_values	
identity??'key_value_init43893/LookupTableImportV2?
'key_value_init43893/LookupTableImportV2LookupTableImportV24key_value_init43893_lookuptableimportv2_table_handle,key_value_init43893_lookuptableimportv2_keys.key_value_init43893_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init43893/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?u:?u2R
'key_value_init43893/LookupTableImportV2'key_value_init43893/LookupTableImportV2:!

_output_shapes	
:?u:!

_output_shapes	
:?u
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_150180

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_151839

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_150204

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_150567

inputsY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	#
embedding_150551:	?u  
dense_2_150555:  
dense_2_150557:  
dense_3_150561: 
dense_3_150563:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2\
 text_vectorization_2/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0embedding_150551*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_150164?
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905?
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_150555dense_2_150557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_150180?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150262?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_150561dense_3_150563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_150204w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_sequential_1_layer_call_fn_150611
text_vectorization_2_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?u 
	unknown_4:  
	unknown_5: 
	unknown_6: 
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_150567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151720

inputsY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	4
!embedding_embedding_lookup_151690:	?u 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding/embedding_lookup?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2\
 text_vectorization_2/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_151690Atext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/151690*+
_output_shapes
:?????????x *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/151690*+
_output_shapes
:?????????x ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? a
dropout_1/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_151792

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
+
__inference_<lambda>_151912
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151447

inputsY
Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_2_string_lookup_2_equal_y3
/text_vectorization_2_string_lookup_2_selectv2_t	4
!embedding_embedding_lookup_151424:	?u 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding/embedding_lookup?Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2\
 text_vectorization_2/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_2/StaticRegexReplaceStaticRegexReplace)text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
)text_vectorization_2/StaticRegexReplace_1StaticRegexReplace0text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
)text_vectorization_2/StaticRegexReplace_2StaticRegexReplace2text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_3StaticRegexReplace2text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
)text_vectorization_2/StaticRegexReplace_4StaticRegexReplace2text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
)text_vectorization_2/StaticRegexReplace_5StaticRegexReplace2text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
)text_vectorization_2/StaticRegexReplace_6StaticRegexReplace2text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
)text_vectorization_2/StaticRegexReplace_7StaticRegexReplace2text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
)text_vectorization_2/StaticRegexReplace_8StaticRegexReplace2text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
)text_vectorization_2/StaticRegexReplace_9StaticRegexReplace2text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
*text_vectorization_2/StaticRegexReplace_10StaticRegexReplace2text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
*text_vectorization_2/StaticRegexReplace_11StaticRegexReplace3text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
*text_vectorization_2/StaticRegexReplace_12StaticRegexReplace3text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_13StaticRegexReplace3text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
*text_vectorization_2/StaticRegexReplace_14StaticRegexReplace3text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
*text_vectorization_2/StaticRegexReplace_15StaticRegexReplace3text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_16StaticRegexReplace3text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_17StaticRegexReplace3text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_18StaticRegexReplace3text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_19StaticRegexReplace3text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_20StaticRegexReplace3text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_21StaticRegexReplace3text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_22StaticRegexReplace3text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_23StaticRegexReplace3text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_24StaticRegexReplace3text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_25StaticRegexReplace3text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_26StaticRegexReplace3text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_27StaticRegexReplace3text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_28StaticRegexReplace3text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_29StaticRegexReplace3text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_30StaticRegexReplace3text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_31StaticRegexReplace3text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_32StaticRegexReplace3text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_33StaticRegexReplace3text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_34StaticRegexReplace3text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_35StaticRegexReplace3text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_36StaticRegexReplace3text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_37StaticRegexReplace3text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_38StaticRegexReplace3text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_39StaticRegexReplace3text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_40StaticRegexReplace3text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_41StaticRegexReplace3text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_42StaticRegexReplace3text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_43StaticRegexReplace3text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_44StaticRegexReplace3text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_45StaticRegexReplace3text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_46StaticRegexReplace3text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_47StaticRegexReplace3text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_48StaticRegexReplace3text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_49StaticRegexReplace3text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_50StaticRegexReplace3text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_51StaticRegexReplace3text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_52StaticRegexReplace3text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_53StaticRegexReplace3text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_54StaticRegexReplace3text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_55StaticRegexReplace3text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_56StaticRegexReplace3text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_57StaticRegexReplace3text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_58StaticRegexReplace3text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_59StaticRegexReplace3text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_60StaticRegexReplace3text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_61StaticRegexReplace3text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_62StaticRegexReplace3text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_63StaticRegexReplace3text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_64StaticRegexReplace3text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_65StaticRegexReplace3text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_66StaticRegexReplace3text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_67StaticRegexReplace3text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_68StaticRegexReplace3text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_69StaticRegexReplace3text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_70StaticRegexReplace3text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_71StaticRegexReplace3text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_72StaticRegexReplace3text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_73StaticRegexReplace3text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_74StaticRegexReplace3text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_75StaticRegexReplace3text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_76StaticRegexReplace3text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_77StaticRegexReplace3text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_78StaticRegexReplace3text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_79StaticRegexReplace3text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_80StaticRegexReplace3text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_81StaticRegexReplace3text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_82StaticRegexReplace3text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_83StaticRegexReplace3text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_84StaticRegexReplace3text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_85StaticRegexReplace3text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_86StaticRegexReplace3text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_87StaticRegexReplace3text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_88StaticRegexReplace3text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_89StaticRegexReplace3text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_90StaticRegexReplace3text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_91StaticRegexReplace3text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_92StaticRegexReplace3text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_93StaticRegexReplace3text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_94StaticRegexReplace3text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_95StaticRegexReplace3text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_96StaticRegexReplace3text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_97StaticRegexReplace3text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_98StaticRegexReplace3text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
*text_vectorization_2/StaticRegexReplace_99StaticRegexReplace3text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_100StaticRegexReplace3text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_101StaticRegexReplace4text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_102StaticRegexReplace4text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_103StaticRegexReplace4text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_104StaticRegexReplace4text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_105StaticRegexReplace4text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_106StaticRegexReplace4text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_107StaticRegexReplace4text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_108StaticRegexReplace4text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_109StaticRegexReplace4text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_110StaticRegexReplace4text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_111StaticRegexReplace4text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_112StaticRegexReplace4text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_113StaticRegexReplace4text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_114StaticRegexReplace4text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_115StaticRegexReplace4text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_116StaticRegexReplace4text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_117StaticRegexReplace4text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_118StaticRegexReplace4text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_119StaticRegexReplace4text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_120StaticRegexReplace4text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_121StaticRegexReplace4text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_122StaticRegexReplace4text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_123StaticRegexReplace4text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_124StaticRegexReplace4text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_125StaticRegexReplace4text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_126StaticRegexReplace4text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_127StaticRegexReplace4text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_128StaticRegexReplace4text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_129StaticRegexReplace4text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_130StaticRegexReplace4text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_131StaticRegexReplace4text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_132StaticRegexReplace4text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_133StaticRegexReplace4text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_134StaticRegexReplace4text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_135StaticRegexReplace4text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_136StaticRegexReplace4text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_137StaticRegexReplace4text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_138StaticRegexReplace4text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_139StaticRegexReplace4text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_140StaticRegexReplace4text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_141StaticRegexReplace4text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_142StaticRegexReplace4text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_143StaticRegexReplace4text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_144StaticRegexReplace4text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_145StaticRegexReplace4text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_146StaticRegexReplace4text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_147StaticRegexReplace4text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_148StaticRegexReplace4text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_149StaticRegexReplace4text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_150StaticRegexReplace4text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_151StaticRegexReplace4text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_152StaticRegexReplace4text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_153StaticRegexReplace4text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_154StaticRegexReplace4text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_155StaticRegexReplace4text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_156StaticRegexReplace4text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_157StaticRegexReplace4text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_158StaticRegexReplace4text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_159StaticRegexReplace4text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_160StaticRegexReplace4text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_161StaticRegexReplace4text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_162StaticRegexReplace4text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_163StaticRegexReplace4text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_164StaticRegexReplace4text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_165StaticRegexReplace4text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_166StaticRegexReplace4text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_167StaticRegexReplace4text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_168StaticRegexReplace4text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_169StaticRegexReplace4text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_170StaticRegexReplace4text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_171StaticRegexReplace4text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_172StaticRegexReplace4text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_173StaticRegexReplace4text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_174StaticRegexReplace4text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_175StaticRegexReplace4text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_176StaticRegexReplace4text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_177StaticRegexReplace4text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_178StaticRegexReplace4text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_179StaticRegexReplace4text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_180StaticRegexReplace4text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_181StaticRegexReplace4text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_182StaticRegexReplace4text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_183StaticRegexReplace4text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_184StaticRegexReplace4text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_185StaticRegexReplace4text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_186StaticRegexReplace4text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_187StaticRegexReplace4text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_188StaticRegexReplace4text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_189StaticRegexReplace4text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_190StaticRegexReplace4text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_191StaticRegexReplace4text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_192StaticRegexReplace4text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
+text_vectorization_2/StaticRegexReplace_193StaticRegexReplace4text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite g
&text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_2/StringSplit/StringSplitV2StringSplitV24text_vectorization_2/StaticRegexReplace_193:output:0/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_2/StringSplit/strided_sliceStridedSlice8text_vectorization_2/StringSplit/StringSplitV2:indices:0=text_vectorization_2/StringSplit/strided_slice/stack:output:0?text_vectorization_2/StringSplit/strided_slice/stack_1:output:0?text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_2/StringSplit/strided_slice_1StridedSlice6text_vectorization_2/StringSplit/StringSplitV2:shape:0?text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_2/StringSplit/StringSplitV2:values:0Vtext_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_2/string_lookup_2/EqualEqual7text_vectorization_2/StringSplit/StringSplitV2:values:0,text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/SelectV2SelectV2.text_vectorization_2/string_lookup_2/Equal:z:0/text_vectorization_2_string_lookup_2_selectv2_tQtext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_2/string_lookup_2/IdentityIdentity6text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
8text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_2/RaggedToTensor/Const:output:06text_vectorization_2/string_lookup_2/Identity:output:0:text_vectorization_2/RaggedToTensor/default_value:output:09text_vectorization_2/StringSplit/strided_slice_1:output:07text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_151424Atext_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/151424*+
_output_shapes
:?????????x *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/151424*+
_output_shapes
:?????????x ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x s
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? l
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_3/MatMulMatMuldropout_1/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding/embedding_lookupI^text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Htext_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?g
?
"__inference__traced_restore_152106
file_prefix8
%assignvariableop_embedding_embeddings:	?u 3
!assignvariableop_1_dense_2_kernel:  -
assignvariableop_2_dense_2_bias: 3
!assignvariableop_3_dense_3_kernel: -
assignvariableop_4_dense_3_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: B
/assignvariableop_14_adam_embedding_embeddings_m:	?u ;
)assignvariableop_15_adam_dense_2_kernel_m:  5
'assignvariableop_16_adam_dense_2_bias_m: ;
)assignvariableop_17_adam_dense_3_kernel_m: 5
'assignvariableop_18_adam_dense_3_bias_m:B
/assignvariableop_19_adam_embedding_embeddings_v:	?u ;
)assignvariableop_20_adam_dense_2_kernel_v:  5
'assignvariableop_21_adam_dense_2_bias_v: ;
)assignvariableop_22_adam_dense_3_kernel_v: 5
'assignvariableop_23_adam_dense_3_bias_v:
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_embedding_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_3_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_3_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_2_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_2_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_3_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_3_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
??
?
!__inference__wrapped_model_149895
text_vectorization_2_inputf
bsequential_1_text_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleg
csequential_1_text_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value	=
9sequential_1_text_vectorization_2_string_lookup_2_equal_y@
<sequential_1_text_vectorization_2_string_lookup_2_selectv2_t	A
.sequential_1_embedding_embedding_lookup_149872:	?u E
3sequential_1_dense_2_matmul_readvariableop_resource:  B
4sequential_1_dense_2_biasadd_readvariableop_resource: E
3sequential_1_dense_3_matmul_readvariableop_resource: B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity??+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?'sequential_1/embedding/embedding_lookup?Usequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2}
-sequential_1/text_vectorization_2/StringLowerStringLowertext_vectorization_2_input*#
_output_shapes
:??????????
4sequential_1/text_vectorization_2/StaticRegexReplaceStaticRegexReplace6sequential_1/text_vectorization_2/StringLower:output:0*#
_output_shapes
:?????????*
patternwon't*
rewrite
will not?
6sequential_1/text_vectorization_2/StaticRegexReplace_1StaticRegexReplace=sequential_1/text_vectorization_2/StaticRegexReplace:output:0*#
_output_shapes
:?????????*
patterncan't*
rewrite	can not?
6sequential_1/text_vectorization_2/StaticRegexReplace_2StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_1:output:0*#
_output_shapes
:?????????*
patternn't*
rewrite not?
6sequential_1/text_vectorization_2/StaticRegexReplace_3StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_2:output:0*#
_output_shapes
:?????????*
pattern're*
rewrite are?
6sequential_1/text_vectorization_2/StaticRegexReplace_4StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_3:output:0*#
_output_shapes
:?????????*
pattern's*
rewrite is?
6sequential_1/text_vectorization_2/StaticRegexReplace_5StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_4:output:0*#
_output_shapes
:?????????*
pattern'd*
rewrite would?
6sequential_1/text_vectorization_2/StaticRegexReplace_6StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_5:output:0*#
_output_shapes
:?????????*
pattern'll*
rewrite will?
6sequential_1/text_vectorization_2/StaticRegexReplace_7StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_6:output:0*#
_output_shapes
:?????????*
pattern't*
rewrite not?
6sequential_1/text_vectorization_2/StaticRegexReplace_8StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_7:output:0*#
_output_shapes
:?????????*
pattern've*
rewrite have?
6sequential_1/text_vectorization_2/StaticRegexReplace_9StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_8:output:0*#
_output_shapes
:?????????*
pattern'm*
rewrite am?
7sequential_1/text_vectorization_2/StaticRegexReplace_10StaticRegexReplace?sequential_1/text_vectorization_2/StaticRegexReplace_9:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_11StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_10:output:0*#
_output_shapes
:?????????*+
pattern \d+(?:\.\d*)?(?:[eE][+-]?\d+)?*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_12StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_11:output:0*#
_output_shapes
:?????????*
pattern@([A-Za-z0-9_]+)*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_13StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_12:output:0*#
_output_shapes
:?????????*
pattern	\([^)]*\)*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_14StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_13:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_15StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_14:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+shan[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_16StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_15:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+i[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_17StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_16:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+what[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_18StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_17:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+few[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_19StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_18:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+that[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_20StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_19:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+into[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_21StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_20:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+needn[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_22StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_21:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+the[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_23StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_22:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+having[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_24StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_23:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+same[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_25StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_24:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+itself[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_26StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_25:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+between[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_27StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_26:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+doesn't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_28StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_27:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+yourselves[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_29StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_28:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+until[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_30StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_29:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+just[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_31StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_30:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+weren[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_32StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_31:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+shouldn[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_33StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_32:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+aren't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_34StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_33:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+below[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_35StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_34:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+as[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_36StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_35:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+had[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_37StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_36:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+other[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_38StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_37:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+no[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_39StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_38:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+didn't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_40StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_39:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+any[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_41StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_40:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+y[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_42StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_41:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doing[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_43StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_42:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+we[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_44StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_43:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+here[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_45StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_44:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+t[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_46StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_45:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+their[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_47StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_46:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+are[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_48StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_47:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hadn[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_49StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_48:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+before[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_50StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_49:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+over[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_51StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_50:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+couldn't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_52StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_51:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+o[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_53StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_52:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+our[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_54StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_53:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+those[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_55StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_54:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+re[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_56StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_55:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+which[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_57StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_56:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+if[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_58StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_57:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+more[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_59StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_58:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+or[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_60StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_59:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+while[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_61StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_60:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+your[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_62StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_61:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+off[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_63StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_62:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+couldn[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_64StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_63:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+so[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_65StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_64:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+during[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_66StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_65:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+be[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_67StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_66:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+once[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_68StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_67:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+now[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_69StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_68:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+of[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_70StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_69:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+not[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_71StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_70:output:0*#
_output_shapes
:?????????*3
pattern(&[^A-Za-z0-9_]+themselves[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_72StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_71:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+under[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_73StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_72:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+from[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_74StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_73:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+by[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_75StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_74:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+they[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_76StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_75:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+she[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_77StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_76:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+mustn't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_78StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_77:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+an[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_79StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_78:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+being[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_80StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_79:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+too[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_81StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_80:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+where[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_82StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_81:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+who[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_83StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_82:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you've[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_84StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_83:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+you[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_85StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_84:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+doesn[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_86StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_85:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+again[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_87StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_86:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+don't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_88StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_87:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+only[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_89StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_88:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+this[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_90StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_89:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+can[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_91StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_90:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+needn't[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_92StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_91:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+my[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_93StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_92:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+up[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_94StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_93:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+down[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_95StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_94:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+in[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_96StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_95:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+to[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_97StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_96:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+yourself[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_98StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_97:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+will[^A-Za-z0-9_]+*
rewrite ?
7sequential_1/text_vectorization_2/StaticRegexReplace_99StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_98:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+myself[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_100StaticRegexReplace@sequential_1/text_vectorization_2/StaticRegexReplace_99:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+herself[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_101StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_100:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+has[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_102StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_101:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+did[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_103StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_102:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+wouldn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_104StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_103:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+a[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_105StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_104:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+m[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_106StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_105:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+them[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_107StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_106:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+her[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_108StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_107:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+these[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_109StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_108:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+it[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_110StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_109:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+were[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_111StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_110:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ve[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_112StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_111:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hasn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_113StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_112:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+have[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_114StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_113:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+haven't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_115StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_114:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+nor[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_116StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_115:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hasn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_117StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_116:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+mightn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_118StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_117:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+how[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_119StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_118:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ma[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_120StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_119:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+its[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_121StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_120:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you'll[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_122StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_121:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+there[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_123StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_122:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+such[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_124StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_123:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+theirs[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_125StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_124:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+been[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_126StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_125:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+am[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_127StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_126:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+at[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_128StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_127:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+with[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_129StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_128:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+hadn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_130StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_129:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+each[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_131StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_130:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+ourselves[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_132StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_131:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+that'll[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_133StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_132:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+shouldn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_134StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_133:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+isn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_135StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_134:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+it's[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_136StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_135:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+didn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_137StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_136:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+both[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_138StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_137:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+and[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_139StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_138:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+because[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_140StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_139:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+after[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_141StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_140:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+his[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_142StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_141:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+should[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_143StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_142:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+very[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_144StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_143:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+for[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_145StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_144:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+above[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_146StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_145:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+haven[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_147StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_146:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+about[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_148StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_147:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+further[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_149StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_148:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+ll[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_150StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_149:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+hers[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_151StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_150:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+d[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_152StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_151:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+me[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_153StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_152:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wasn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_154StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_153:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+he[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_155StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_154:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+shan't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_156StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_155:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+then[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_157StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_156:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+him[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_158StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_157:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+don[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_159StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_158:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+yours[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_160StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_159:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+she's[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_161StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_160:output:0*#
_output_shapes
:?????????*2
pattern'%[^A-Za-z0-9_]+should've[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_162StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_161:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+some[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_163StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_162:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+weren't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_164StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_163:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+won't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_165StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_164:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+than[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_166StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_165:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+is[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_167StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_166:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+why[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_168StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_167:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+was[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_169StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_168:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+whom[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_170StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_169:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+through[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_171StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_170:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+out[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_172StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_171:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+ain[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_173StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_172:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+on[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_174StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_173:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+all[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_175StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_174:output:0*#
_output_shapes
:?????????*1
pattern&$[^A-Za-z0-9_]+mightn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_176StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_175:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+you'd[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_177StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_176:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+but[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_178StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_177:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+wouldn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_179StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_178:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+mustn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_180StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_179:output:0*#
_output_shapes
:?????????*/
pattern$"[^A-Za-z0-9_]+you're[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_181StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_180:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+own[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_182StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_181:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+against[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_183StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_182:output:0*#
_output_shapes
:?????????**
pattern[^A-Za-z0-9_]+s[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_184StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_183:output:0*#
_output_shapes
:?????????*.
pattern#![^A-Za-z0-9_]+isn't[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_185StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_184:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+wasn[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_186StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_185:output:0*#
_output_shapes
:?????????*,
pattern![^A-Za-z0-9_]+won[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_187StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_186:output:0*#
_output_shapes
:?????????*0
pattern%#[^A-Za-z0-9_]+himself[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_188StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_187:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+does[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_189StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_188:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+when[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_190StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_189:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+ours[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_191StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_190:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+most[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_192StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_191:output:0*#
_output_shapes
:?????????*+
pattern [^A-Za-z0-9_]+do[^A-Za-z0-9_]+*
rewrite ?
8sequential_1/text_vectorization_2/StaticRegexReplace_193StaticRegexReplaceAsequential_1/text_vectorization_2/StaticRegexReplace_192:output:0*#
_output_shapes
:?????????*-
pattern" [^A-Za-z0-9_]+aren[^A-Za-z0-9_]+*
rewrite t
3sequential_1/text_vectorization_2/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
;sequential_1/text_vectorization_2/StringSplit/StringSplitV2StringSplitV2Asequential_1/text_vectorization_2/StaticRegexReplace_193:output:0<sequential_1/text_vectorization_2/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Asequential_1/text_vectorization_2/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Csequential_1/text_vectorization_2/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Csequential_1/text_vectorization_2/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
;sequential_1/text_vectorization_2/StringSplit/strided_sliceStridedSliceEsequential_1/text_vectorization_2/StringSplit/StringSplitV2:indices:0Jsequential_1/text_vectorization_2/StringSplit/strided_slice/stack:output:0Lsequential_1/text_vectorization_2/StringSplit/strided_slice/stack_1:output:0Lsequential_1/text_vectorization_2/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Csequential_1/text_vectorization_2/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_1/text_vectorization_2/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential_1/text_vectorization_2/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential_1/text_vectorization_2/StringSplit/strided_slice_1StridedSliceCsequential_1/text_vectorization_2/StringSplit/StringSplitV2:shape:0Lsequential_1/text_vectorization_2/StringSplit/strided_slice_1/stack:output:0Nsequential_1/text_vectorization_2/StringSplit/strided_slice_1/stack_1:output:0Nsequential_1/text_vectorization_2/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
dsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastDsequential_1/text_vectorization_2/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
fsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastFsequential_1/text_vectorization_2/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
nsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapehsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
nsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
msequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdwsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0wsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
rsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatervsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0{sequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
msequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasttsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
lsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxhsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ysequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
lsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2usequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0wsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
lsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulqsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumjsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumjsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0tsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
qsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounthsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ysequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
ksequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumxsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0tsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
osequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
ksequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2xsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0lsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0tsequential_1/text_vectorization_2/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Usequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2LookupTableFindV2bsequential_1_text_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_table_handleDsequential_1/text_vectorization_2/StringSplit/StringSplitV2:values:0csequential_1_text_vectorization_2_string_lookup_2_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
7sequential_1/text_vectorization_2/string_lookup_2/EqualEqualDsequential_1/text_vectorization_2/StringSplit/StringSplitV2:values:09sequential_1_text_vectorization_2_string_lookup_2_equal_y*
T0*#
_output_shapes
:??????????
:sequential_1/text_vectorization_2/string_lookup_2/SelectV2SelectV2;sequential_1/text_vectorization_2/string_lookup_2/Equal:z:0<sequential_1_text_vectorization_2_string_lookup_2_selectv2_t^sequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
:sequential_1/text_vectorization_2/string_lookup_2/IdentityIdentityCsequential_1/text_vectorization_2/string_lookup_2/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
>sequential_1/text_vectorization_2/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
6sequential_1/text_vectorization_2/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
Esequential_1/text_vectorization_2/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor?sequential_1/text_vectorization_2/RaggedToTensor/Const:output:0Csequential_1/text_vectorization_2/string_lookup_2/Identity:output:0Gsequential_1/text_vectorization_2/RaggedToTensor/default_value:output:0Fsequential_1/text_vectorization_2/StringSplit/strided_slice_1:output:0Dsequential_1/text_vectorization_2/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
'sequential_1/embedding/embedding_lookupResourceGather.sequential_1_embedding_embedding_lookup_149872Nsequential_1/text_vectorization_2/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding/embedding_lookup/149872*+
_output_shapes
:?????????x *
dtype0?
0sequential_1/embedding/embedding_lookup/IdentityIdentity0sequential_1/embedding/embedding_lookup:output:0*
T0*A
_class7
53loc:@sequential_1/embedding/embedding_lookup/149872*+
_output_shapes
:?????????x ?
2sequential_1/embedding/embedding_lookup/Identity_1Identity9sequential_1/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x ?
>sequential_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential_1/global_average_pooling1d_1/MeanMean;sequential_1/embedding/embedding_lookup/Identity_1:output:0Gsequential_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
sequential_1/dense_2/MatMulMatMul5sequential_1/global_average_pooling1d_1/Mean:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? z
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
sequential_1/dropout_1/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp(^sequential_1/embedding/embedding_lookupV^sequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2R
'sequential_1/embedding/embedding_lookup'sequential_1/embedding/embedding_lookup2?
Usequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2Usequential_1/text_vectorization_2/string_lookup_2/hash_table_Lookup/LookupTableFindV2:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

*__inference_embedding_layer_call_fn_151752

inputs	
unknown:	?u 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_150164s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????x `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
c
*__inference_dropout_1_layer_call_fn_151802

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_150262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_1_layer_call_fn_151766

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_149905i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
]
text_vectorization_2_input?
,serving_default_text_vectorization_2_input:0?????????=
dense_32
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
P
_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,_random_generator
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemp mq!mr/ms0mtvu vv!vw/vx0vy"
	optimizer
C
1
 2
!3
/4
05"
trackable_list_wrapper
C
0
 1
!2
/3
04"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_sequential_1_layer_call_fn_150232
-__inference_sequential_1_layer_call_fn_151158
-__inference_sequential_1_layer_call_fn_151181
-__inference_sequential_1_layer_call_fn_150611?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151447
H__inference_sequential_1_layer_call_and_return_conditional_losses_151720
H__inference_sequential_1_layer_call_and_return_conditional_losses_150870
H__inference_sequential_1_layer_call_and_return_conditional_losses_151129?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_149895text_vectorization_2_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Aserving_default"
signature_map
L
Blookup_table
Ctoken_counts
D	keras_api"
_tf_keras_layer
"
_generic_user_object
?2?
__inference_adapt_step_149148?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%	?u 2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_embedding_layer_call_fn_151752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_layer_call_and_return_conditional_losses_151761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
;__inference_global_average_pooling1d_1_layer_call_fn_151766?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_151772?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :  2dense_2/kernel
: 2dense_2/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_2_layer_call_fn_151781?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_151792?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
*__inference_dropout_1_layer_call_fn_151797
*__inference_dropout_1_layer_call_fn_151802?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_151807
E__inference_dropout_1_layer_call_and_return_conditional_losses_151819?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 : 2dense_3/kernel
:2dense_3/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_3_layer_call_fn_151828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_151839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_151745text_vectorization_2_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
`_initializer
a_create_resource
b_initialize
c_destroy_resourceR jCustom.StaticHashTable
O
d_create_resource
e_initialize
f_destroy_resourceR Z
tablez{
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	gtotal
	hcount
i	variables
j	keras_api"
_tf_keras_metric
^
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api"
_tf_keras_metric
"
_generic_user_object
?2?
__inference__creator_151844?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_151852?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_151857?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_151862?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_151867?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_151872?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
:  (2total
:  (2count
.
g0
h1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
,:*	?u 2Adam/embedding/embeddings/m
%:#  2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
%:# 2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
,:*	?u 2Adam/embedding/embeddings/v
%:#  2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
%:# 2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
?B?
__inference_save_fn_151891checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_151899restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_57
__inference__creator_151844?

? 
? "? 7
__inference__creator_151862?

? 
? "? 9
__inference__destroyer_151857?

? 
? "? 9
__inference__destroyer_151872?

? 
? "? B
__inference__initializer_151852B???

? 
? "? ;
__inference__initializer_151867?

? 
? "? ?
!__inference__wrapped_model_149895	B|}~ !/0??<
5?2
0?-
text_vectorization_2_input?????????
? "1?.
,
dense_3!?
dense_3?????????j
__inference_adapt_step_149148IC??<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
C__inference_dense_2_layer_call_and_return_conditional_losses_151792\ !/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? {
(__inference_dense_2_layer_call_fn_151781O !/?,
%?"
 ?
inputs????????? 
? "?????????? ?
C__inference_dense_3_layer_call_and_return_conditional_losses_151839\/0/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_151828O/0/?,
%?"
 ?
inputs????????? 
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_151807\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_151819\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? }
*__inference_dropout_1_layer_call_fn_151797O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? }
*__inference_dropout_1_layer_call_fn_151802O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
E__inference_embedding_layer_call_and_return_conditional_losses_151761_/?,
%?"
 ?
inputs?????????x	
? ")?&
?
0?????????x 
? ?
*__inference_embedding_layer_call_fn_151752R/?,
%?"
 ?
inputs?????????x	
? "??????????x ?
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_151772{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
;__inference_global_average_pooling1d_1_layer_call_fn_151766nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????z
__inference_restore_fn_151899YCK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_151891?C&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
H__inference_sequential_1_layer_call_and_return_conditional_losses_150870{	B|}~ !/0G?D
=?:
0?-
text_vectorization_2_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151129{	B|}~ !/0G?D
=?:
0?-
text_vectorization_2_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151447g	B|}~ !/03?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_151720g	B|}~ !/03?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_1_layer_call_fn_150232n	B|}~ !/0G?D
=?:
0?-
text_vectorization_2_input?????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_150611n	B|}~ !/0G?D
=?:
0?-
text_vectorization_2_input?????????
p

 
? "???????????
-__inference_sequential_1_layer_call_fn_151158Z	B|}~ !/03?0
)?&
?
inputs?????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_151181Z	B|}~ !/03?0
)?&
?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_151745?	B|}~ !/0]?Z
? 
S?P
N
text_vectorization_2_input0?-
text_vectorization_2_input?????????"1?.
,
dense_3!?
dense_3?????????