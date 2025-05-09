#ifndef CLUSTER2_H_
#define CLUSTER2_H_
#include <string>
#include <vector>
#include <list>
#include <map>
#include <list>
#include "Clone.h"
#include "LCS2.h"
#include "EditDistance.h"

class Cluster2 {
	std::string original;
	std::vector<Clone> clones;
	std::vector<Clone> clonesBackup;
	std::vector<std::vector<LetterOps> > editLists;
	std::vector<LCS2> lcs2s;
public:
	Cluster2(const unsigned strandLength, const int clonesNum, const double delProb, const double insProb,
			const double subProb, std::mt19937& generator);
	Cluster2(const std::string& original, const std::vector<std::string>& copies);
	void CmpLCS2(const unsigned int anchorIndex);
	void ComputeEditLists(const unsigned int anchorIndex, const int priority, std::mt19937& generator, char* deviceX, char* deviceY,
			int* dp, int* deviceArr);
	void ReverseClones();
	std::string Original() const;
	std::vector<int> InsArray(const int cloneIndex);
	std::vector<std::map<std::string, int> > CumEditOperationsNoInserts(const int cloneIndex, const int priority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::map<std::string, int> > CumEditOperationsInserts(const int cloneIndex, const int priority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::map<std::string, int> > RepStringLowercaseGapsDictionaries(const int cloneIndex,
			const int patternLen);
	std::vector<std::map<std::string, int> > RepStringRDIDictionaries(const int cloneIndex, const int patternLen,
			const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::map<std::string, int> > RepStringRDISubDictionaries(const int cloneIndex, const int patternLen,
			const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::map<std::string, int> > RepStringRDISubNoGapsDictionaries(const int cloneIndex,
			const int patternLen, const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::map<std::string, int> > RepStringNumberGapsDictionaries(const int cloneIndex,
			const int patternLen);
	std::vector<std::map<std::string, int> > RepStringNoGapsDictionaries(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertByInsArray(const int cloneIndex, double ins_threshold);
	std::vector<int> GuessInsertsByLowercaseGapsStringMax(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertsByLowercaseGapsStringPath(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertsByNumberGapsStringMax(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertsByNumberGapsStringPath(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertsByNoGapsStringMax(const int cloneIndex, const int patternLen);
	std::vector<int> GuessInsertsByNoGapsStringPath(const int cloneIndex, const int patternLen);
	std::vector<std::pair<int, std::string> > GuessDeletionsByLowercaseGapsStringMax(const int cloneIndex,
			const int patternLen);
	std::vector<std::pair<int, std::string> > GuessDeletionsByLowercaseGapsStringPath(const int cloneIndex,
			const int patternLen);
	std::vector<std::pair<int, std::string> > GuessDeletionsByRDIStringPath(const int cloneIndex, const int patternLen,
			const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::pair<int, std::string> > GuessSubstitutions(const int cloneIndex, const int priority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::pair<int, std::string> > GuessSubstitutionsRDIPath(const int cloneIndex, const int patternLen,
			const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::pair<int, std::string> > GuessSubstitutionsRDINoGapsPath(const int cloneIndex,
			const int patternLen, const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<int> GuessInsertsRDI(const int cloneIndex, const int priority, std::mt19937& generator,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::vector<std::pair<int, std::string> > GuessDeletionsRDI(const int cloneIndex, const int priority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllDeletions(const int patternLen);
	void FixAllInserts(const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllSubstitutions(const int subPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllDeletionsClonesLastInitial(const int delPatternLen, const int delPriority, std::mt19937& generator);
	void FixAllInsertsClonesLastInitial(const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllSubstitutionsClonesLastInitial(const int subPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesLastInitialNormal(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesLastInitialNormalAndReverse(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesVerticalHorizontalNormal(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesVerticalHorizontalNormalAndReverse(const int delPatternLen, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int maxReps,
			std::vector<Clone>& normalResult, std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesHorizontalVerticalNormal(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesHorizontalVerticalNormalAndReverse(const int delPatternLen, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int maxReps,
			std::vector<Clone>& normalResult, std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixOneCopyRDIAll(const int cloneIndex, const int subPatternLen, const int delPatternLen,
			const double insThreshold, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixOneCopyRDIDelPath(const int cloneIndex, const int delPatternLen, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, 
			int* dp, int* deviceArr);
	void FixAllOneByOneRDI(const int delPatternLen, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllNormal(const int delPatternLen, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllOneByOneRDIChangeClonesLast(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllOneByOneRDIChangeClonesLastInitial(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllOneByOneRDIRepeat(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllOneByOneRDIChangeClonesLastRepeat(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int minEqual, const int maxReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllOneByOneRDIChangeClonesLastInitialRepeat(const int delPatternLen, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int maxReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void SortClonesByInitial(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestInsertGuesses(const int cloneIndex, const double insThreshold, const int patternLen,
			int* trueInsertGuesses, int* falseInsertGuesses, int* totalInserts, int* guessesNum);
	void TestDeletionGuesses(const int cloneIndex, const int patternLen, int* trueDeletionGuesses,
			int* falseDeletionGuesses, int* totalDeletions, int* guessesNum);
	void TestSubstitutionGuesses(const int clone_index, std::map<std::string, double>& roundCountBefore,
			std::map<std::string, double>& roundCountAfter, double& distNoReplaceBefore, double& distNoReplaceAfter,
			const int priority, std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAll(const int delPatternLen, std::vector<int>& cloneOriginalEDS, double& avgCloneEditDist,
			int& cumCorrectSizeCloneEditDist, int& correctSizeCloneNum, int& bingoNum, int& firstHalfBingoNum,
			int& secondHalfBingoNum, int& minED, int& maxED, int& finalGuessEditDist, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int maxReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAllClonesLast(const int delPatternLen, std::vector<int>& cloneOriginalEDS, double& avgCloneEditDist,
			int& cumCorrectSizeCloneEditDist, int& correctSizeCloneNum, int& bingoNum, int& firstHalfBingoNum,
			int& secondHalfBingoNum, int& minED, int& maxED, int& finalGuessEditDist, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int minEqual,
			const int maxReps, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllClonesLastNormal(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int minEqual, const int maxReps,
			std::vector<Clone>& normalResult, std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAllClonesLastInitial(const int delPatternLen, std::vector<int>& cloneOriginalEDS,
			double& avgCloneEditDist, int& cumCorrectSizeCloneEditDist, int& correctSizeCloneNum, int& bingoNum,
			int& firstHalfBingoNum, int& secondHalfBingoNum, int& minED, int& maxED, int& finalGuessEditDist,
			const int subPriority, const int delPriority, const int insPriority, std::mt19937& generator,
			const int maxReps, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAllClonesLastInitialMixed(const int subPatternLen, const int delPatternLen, const double insThreshold,
			std::vector<int>& cloneOriginalEDS, double& avgCloneEditDist, int& cumCorrectSizeCloneEditDist,
			int& correctSizeCloneNum, int& bingoNum, int& minED, int& maxED, int& finalGuessEditDist,
			int& finalGuessNum, int& finalStage, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, const int minEqual, const int maxReps);
	void FixAllByErrorTypeRepeat(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllByErrorTypeNormal(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllByErrorTypeNormalAndReverse(const int delPatternLen, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, std::vector<Clone>& normalResult,
			std::vector<Clone>& reverseResult, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAllByErrorType(const int delPatternLen, const double insThreshold, std::vector<int>& cloneOriginalEDS,
			double& avgCloneEditDist, int& cumCorrectSizeCloneEditDist, int& correctSizeCloneNum, int& bingoNum,
			int& firstHalfBingoNum, int& secondHalfBingoNum, int& minED, int& maxED, int& finalGuessEditDist,
			const int subPriority, const int delPriority, const int insPriority, std::mt19937& generator,
			const int minEqual, const int maxReps, const int convReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void FixAllMixed(const int delPatternLen, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, const int maxReps, char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestFixAllMixed(const int delPatternLen, std::vector<int>& cloneOriginalEDS, double& avgCloneEditDist,
			int& cumCorrectSizeCloneEditDist, int& correctSizeCloneNum, int& bingoNum, int& firstHalfbingoNum,
			int& secondHalfbingoNum, int& minED, int& maxED, int& finalGuessEditDist, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator, const int maxReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	std::string TestBest(const int delPatternLen, int& finalGuessEditDist, const int subPriority, const int delPriority,
			const int insPriority, std::mt19937& generator, const int maxReps,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	void TestOriginalRetention(const int delPatternLen, int& finalGuessEditDist, const int subPriority,
			const int delPriority, const int insPriority, std::mt19937& generator,
			char* deviceX, char* deviceY, int* dp, int* deviceArr);
	int GetInsertNum(const int cloneIndex) const;
	int GetDeleteNum(const int cloneIndex) const;
	void CloneStats(std::vector<int>& lengths, std::vector<int>& originalED, std::vector<int>& clonesED) const;
	void PrintStats() const;
	void Stats(const int delPatternLen, const int subPriority, const int delPriority, const int insPriority,
			std::mt19937& generator, char* deviceX, char* deviceY, int* dp, int* deviceArr);
};

#endif /* CLUSTER2_H_ */
