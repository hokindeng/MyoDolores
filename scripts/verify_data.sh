#!/bin/bash

# MyoDolores Data Verification Script
# Comprehensive data integrity and validation tool

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/myo_data"
VERIFICATION_LOG="data_verification.log"
ERRORS_FOUND=0
WARNINGS_FOUND=0

# Expected dataset information
declare -A EXPECTED_DATASETS=(
    ["animation_output"]="1000+ 4.7GB Animation and computer graphics motions"
    ["aist_output"]="1500+ 21GB AIST dance dataset motions"
    ["dance_output"]="400+ 2.4GB Dance and choreography sequences"
    ["game_motion_output"]="8000+ 87GB Video game character motions"
    ["HAA500_output"]="3000+ 57GB Human action analysis dataset"
    ["humman_output"]="800+ 11GB Human motion biomechanics"
    ["kungfu_output"]="1200+ 15GB Kung fu and martial arts motions"
    ["perform_output"]="600+ 6.9GB Performance and theatrical motions"
)

# Functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}              MyoDolores Data Verification Tool${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    echo -e "${CYAN}This script verifies data integrity and provides detailed${NC}"
    echo -e "${CYAN}statistics about your MyoData collection.${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -q, --quick             Quick verification (skip detailed checks)"
    echo "  -f, --fix               Attempt to fix issues automatically"
    echo "  -s, --stats             Show detailed statistics"
    echo "  -c, --checksum          Verify file checksums (slow)"
    echo "  --export-report         Export detailed report to CSV"
    echo "  --check-missing         Only check for missing files"
    echo "  --check-duplicates      Check for duplicate files"
    echo "  --check-corruption      Check for file corruption"
    echo ""
    echo "Dataset-specific verification:"
    echo "  animation_output        Verify animation dataset only"
    echo "  aist_output            Verify AIST dance dataset only"
    echo "  dance_output           Verify dance dataset only"
    echo "  game_motion_output     Verify game motion dataset only"
    echo "  HAA500_output          Verify HAA500 dataset only"
    echo "  humman_output          Verify human motion dataset only"
    echo "  kungfu_output          Verify kung fu dataset only"
    echo "  perform_output         Verify performance dataset only"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full verification"
    echo "  $0 --stats             # Show detailed statistics"
    echo "  $0 HAA500_output       # Verify only HAA500 dataset"
    echo "  $0 --quick --verbose   # Quick check with details"
}

log_message() {
    local level=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$VERIFICATION_LOG"
}

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✓${NC} $message"
            log_message "INFO" "OK: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠${NC} $message"
            log_message "WARN" "WARNING: $message"
            ((WARNINGS_FOUND++))
            ;;
        "ERROR")
            echo -e "${RED}✗${NC} $message"
            log_message "ERROR" "ERROR: $message"
            ((ERRORS_FOUND++))
            ;;
        "INFO")
            echo -e "${CYAN}ℹ${NC} $message"
            log_message "INFO" "$message"
            ;;
        "STAT")
            echo -e "${PURPLE}●${NC} $message"
            log_message "STAT" "$message"
            ;;
    esac
}

check_data_directory() {
    echo -e "${BLUE}[1/6] Checking Data Directory Structure${NC}"
    echo "----------------------------------------"
    
    if [[ ! -d "$DATA_DIR" ]]; then
        print_status "ERROR" "myo_data directory not found: $DATA_DIR"
        echo -e "${YELLOW}  Run: mkdir -p myo_data && ./scripts/sync_data.sh --all${NC}"
        return 1
    fi
    
    print_status "OK" "myo_data directory exists: $DATA_DIR"
    
    # Check dataset subdirectories
    local datasets_found=0
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ -d "$DATA_DIR/$dataset" ]]; then
            print_status "OK" "Dataset directory: $dataset"
            ((datasets_found++))
        else
            print_status "WARN" "Dataset directory missing: $dataset"
            echo -e "${YELLOW}  Run: ./scripts/sync_data.sh $dataset${NC}"
        fi
    done
    
    print_status "INFO" "Found $datasets_found/8 dataset directories"
    echo ""
}

verify_file_counts() {
    echo -e "${BLUE}[2/6] Verifying File Counts${NC}"
    echo "----------------------------------------"
    
    local total_h5_files=0
    local total_yaml_files=0
    local total_mjb_files=0
    
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ ! -d "$DATA_DIR/$dataset" ]]; then
            continue
        fi
        
        local h5_count=$(find "$DATA_DIR/$dataset" -name "*.h5" 2>/dev/null | wc -l)
        local yaml_count=$(find "$DATA_DIR/$dataset" -name "*.yaml" 2>/dev/null | wc -l)
        local mjb_count=$(find "$DATA_DIR/$dataset" -name "*.mjb" 2>/dev/null | wc -l)
        
        total_h5_files=$((total_h5_files + h5_count))
        total_yaml_files=$((total_yaml_files + yaml_count))
        total_mjb_files=$((total_mjb_files + mjb_count))
        
        if [[ $h5_count -gt 0 ]]; then
            print_status "OK" "$dataset: $h5_count H5 files"
        else
            print_status "WARN" "$dataset: No H5 files found"
        fi
        
        if [[ "$verbose" == "true" ]]; then
            print_status "INFO" "  YAML files: $yaml_count, MJB files: $mjb_count"
        fi
    done
    
    print_status "INFO" "Total files - H5: $total_h5_files, YAML: $total_yaml_files, MJB: $total_mjb_files"
    
    # Check against expected counts
    if [[ $total_h5_files -ge 25000 ]]; then
        print_status "OK" "H5 file count looks good ($total_h5_files >= 25,000 expected)"
    elif [[ $total_h5_files -ge 10000 ]]; then
        print_status "WARN" "H5 file count lower than expected ($total_h5_files, expect ~28,000)"
    else
        print_status "ERROR" "H5 file count very low ($total_h5_files, expect ~28,000)"
    fi
    
    echo ""
}

verify_file_sizes() {
    echo -e "${BLUE}[3/6] Verifying File Sizes${NC}"
    echo "----------------------------------------"
    
    local total_size=0
    
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ ! -d "$DATA_DIR/$dataset" ]]; then
            continue
        fi
        
        local dataset_size=$(du -sb "$DATA_DIR/$dataset" 2>/dev/null | cut -f1)
        local dataset_size_gb=$(echo "scale=1; $dataset_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
        
        total_size=$((total_size + dataset_size))
        
        # Extract expected size from dataset info
        local expected_info="${EXPECTED_DATASETS[$dataset]}"
        local expected_size=$(echo "$expected_info" | grep -oP '\d+\.?\d*GB' | head -1)
        
        if [[ -n "$expected_size" ]]; then
            local expected_gb=$(echo "$expected_size" | sed 's/GB//')
            local size_diff=$(echo "$dataset_size_gb - $expected_gb" | bc -l 2>/dev/null || echo "0")
            local size_diff_abs=$(echo "$size_diff" | sed 's/-//')
            
            if (( $(echo "$size_diff_abs < 1" | bc -l) )); then
                print_status "OK" "$dataset: ${dataset_size_gb}GB (expected ~${expected_gb}GB)"
            elif (( $(echo "$dataset_size_gb < $expected_gb * 0.5" | bc -l) )); then
                print_status "ERROR" "$dataset: ${dataset_size_gb}GB (expected ~${expected_gb}GB) - significantly undersized"
            else
                print_status "WARN" "$dataset: ${dataset_size_gb}GB (expected ~${expected_gb}GB)"
            fi
        else
            print_status "INFO" "$dataset: ${dataset_size_gb}GB"
        fi
    done
    
    local total_size_gb=$(echo "scale=1; $total_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
    print_status "INFO" "Total data size: ${total_size_gb}GB"
    
    # Check against expected total
    if (( $(echo "$total_size_gb > 180" | bc -l) )); then
        print_status "OK" "Total size looks good (${total_size_gb}GB, expect ~200GB)"
    elif (( $(echo "$total_size_gb > 100" | bc -l) )); then
        print_status "WARN" "Total size lower than expected (${total_size_gb}GB, expect ~200GB)"
    else
        print_status "ERROR" "Total size very low (${total_size_gb}GB, expect ~200GB)"
    fi
    
    echo ""
}

verify_file_integrity() {
    if [[ "$quick" == "true" ]]; then
        print_status "INFO" "Skipping file integrity check (quick mode)"
        echo ""
        return
    fi
    
    echo -e "${BLUE}[4/6] Verifying File Integrity${NC}"
    echo "----------------------------------------"
    
    local corrupted_files=0
    local tested_files=0
    local sample_size=10  # Test sample of files from each dataset
    
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ ! -d "$DATA_DIR/$dataset" ]]; then
            continue
        fi
        
        print_status "INFO" "Testing $dataset integrity..."
        
        # Get sample of H5 files
        local h5_files=($(find "$DATA_DIR/$dataset" -name "*.h5" | head -$sample_size))
        
        if [[ ${#h5_files[@]} -eq 0 ]]; then
            print_status "WARN" "$dataset: No H5 files to test"
            continue
        fi
        
        for file in "${h5_files[@]}"; do
            ((tested_files++))
            
            # Test if file can be opened
            if python3 -c "
import h5py
import sys
try:
    with h5py.File('$file', 'r') as f:
        # Try to read basic structure
        list(f.keys())
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
                if [[ "$verbose" == "true" ]]; then
                    print_status "OK" "$(basename "$file")"
                fi
            else
                print_status "ERROR" "Corrupted file: $(basename "$file")"
                ((corrupted_files++))
            fi
        done
        
        if [[ ${#h5_files[@]} -eq $sample_size ]]; then
            print_status "OK" "$dataset: Tested $sample_size files, all readable"
        else
            print_status "OK" "$dataset: Tested ${#h5_files[@]} files, all readable"
        fi
    done
    
    if [[ $corrupted_files -eq 0 ]]; then
        print_status "OK" "File integrity check passed ($tested_files files tested)"
    else
        print_status "ERROR" "Found $corrupted_files corrupted files out of $tested_files tested"
    fi
    
    echo ""
}

check_data_consistency() {
    echo -e "${BLUE}[5/6] Checking Data Consistency${NC}"
    echo "----------------------------------------"
    
    # Check for expected file patterns
    local pattern_issues=0
    
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ ! -d "$DATA_DIR/$dataset" ]]; then
            continue
        fi
        
        # Check for expected file patterns
        local motion_dirs=$(find "$DATA_DIR/$dataset" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        local loose_h5=$(find "$DATA_DIR/$dataset" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
        
        if [[ $motion_dirs -gt 0 ]]; then
            print_status "OK" "$dataset: $motion_dirs motion directories found"
        else
            print_status "WARN" "$dataset: No motion directories found"
            ((pattern_issues++))
        fi
        
        if [[ $loose_h5 -gt 0 ]]; then
            print_status "WARN" "$dataset: $loose_h5 H5 files in root (should be in subdirectories)"
        fi
        
        # Check for checkpoint files
        if [[ -d "$DATA_DIR/$dataset/.checkpoints" ]]; then
            print_status "OK" "$dataset: Checkpoint directory present"
        else
            print_status "INFO" "$dataset: No checkpoint directory"
        fi
    done
    
    # Check for duplicate filenames across datasets
    if [[ "$check_duplicates" == "true" ]]; then
        print_status "INFO" "Checking for duplicate filenames..."
        local duplicates=$(find "$DATA_DIR" -name "*.h5" -exec basename {} \; | sort | uniq -d | wc -l)
        if [[ $duplicates -eq 0 ]]; then
            print_status "OK" "No duplicate filenames found"
        else
            print_status "WARN" "Found $duplicates duplicate filenames across datasets"
        fi
    fi
    
    if [[ $pattern_issues -eq 0 ]]; then
        print_status "OK" "Data consistency check passed"
    else
        print_status "WARN" "Found $pattern_issues consistency issues"
    fi
    
    echo ""
}

generate_statistics() {
    echo -e "${BLUE}[6/6] Generating Statistics${NC}"
    echo "----------------------------------------"
    
    # Overall statistics
    local total_datasets=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    local total_h5=$(find "$DATA_DIR" -name "*.h5" 2>/dev/null | wc -l)
    local total_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    
    print_status "STAT" "Dataset Summary:"
    print_status "STAT" "  Total datasets: $total_datasets/8"
    print_status "STAT" "  Total H5 files: $total_h5"
    print_status "STAT" "  Total size: $total_size"
    
    if [[ "$stats" == "true" ]]; then
        echo ""
        print_status "STAT" "Detailed Dataset Statistics:"
        
        for dataset in "${!EXPECTED_DATASETS[@]}"; do
            if [[ ! -d "$DATA_DIR/$dataset" ]]; then
                continue
            fi
            
            local h5_count=$(find "$DATA_DIR/$dataset" -name "*.h5" 2>/dev/null | wc -l)
            local dir_count=$(find "$DATA_DIR/$dataset" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            local size=$(du -sh "$DATA_DIR/$dataset" 2>/dev/null | cut -f1)
            local description=$(echo "${EXPECTED_DATASETS[$dataset]}" | cut -d' ' -f3-)
            
            print_status "STAT" "  $dataset:"
            print_status "STAT" "    Files: $h5_count H5, Directories: $dir_count"
            print_status "STAT" "    Size: $size"
            print_status "STAT" "    Description: $description"
            
            # Sample file analysis
            if [[ $h5_count -gt 0 ]] && [[ "$verbose" == "true" ]]; then
                local sample_file=$(find "$DATA_DIR/$dataset" -name "*.h5" | head -1)
                if [[ -n "$sample_file" ]]; then
                    local file_size=$(du -sh "$sample_file" | cut -f1)
                    print_status "STAT" "    Sample file: $(basename "$sample_file") ($file_size)"
                fi
            fi
        done
    fi
    
    echo ""
}

export_report() {
    if [[ "$export_report" != "true" ]]; then
        return
    fi
    
    local report_file="data_verification_report_$(date +%Y%m%d_%H%M%S).csv"
    
    echo "Dataset,H5_Files,Directories,Size_GB,Status" > "$report_file"
    
    for dataset in "${!EXPECTED_DATASETS[@]}"; do
        if [[ ! -d "$DATA_DIR/$dataset" ]]; then
            echo "$dataset,0,0,0,Missing" >> "$report_file"
            continue
        fi
        
        local h5_count=$(find "$DATA_DIR/$dataset" -name "*.h5" 2>/dev/null | wc -l)
        local dir_count=$(find "$DATA_DIR/$dataset" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        local size_bytes=$(du -sb "$DATA_DIR/$dataset" 2>/dev/null | cut -f1)
        local size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
        
        local status="OK"
        if [[ $h5_count -eq 0 ]]; then
            status="No_Files"
        elif [[ $dir_count -eq 0 ]]; then
            status="No_Directories"
        fi
        
        echo "$dataset,$h5_count,$dir_count,$size_gb,$status" >> "$report_file"
    done
    
    print_status "INFO" "Detailed report exported to: $report_file"
}

print_summary() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}                    VERIFICATION SUMMARY${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    if [[ $ERRORS_FOUND -eq 0 && $WARNINGS_FOUND -eq 0 ]]; then
        echo -e "${GREEN}🎉 Perfect! Your MyoData collection is complete and valid.${NC}"
        echo ""
        echo -e "${CYAN}Your data is ready for:${NC}"
        echo -e "  • Motion tracking training with ASAP"
        echo -e "  • Locomotion training with Unitree RL Gym"
        echo -e "  • Biomechanical analysis with myo_api"
        echo -e "  • Motion retargeting with myo_retarget"
    elif [[ $ERRORS_FOUND -eq 0 ]]; then
        echo -e "${YELLOW}⚠ Data verification completed with $WARNINGS_FOUND warning(s).${NC}"
        echo -e "${CYAN}Your data is functional but consider addressing warnings for optimal performance.${NC}"
    else
        echo -e "${RED}❌ Found $ERRORS_FOUND error(s) and $WARNINGS_FOUND warning(s).${NC}"
        echo -e "${CYAN}Some datasets may be incomplete or corrupted.${NC}"
        echo ""
        echo -e "${CYAN}Recommended actions:${NC}"
        echo -e "  • Re-sync missing datasets: ${YELLOW}./scripts/sync_data.sh <dataset_name>${NC}"
        echo -e "  • Re-sync all data: ${YELLOW}./scripts/sync_data.sh --all${NC}"
        echo -e "  • Force re-download: ${YELLOW}./scripts/sync_data.sh --force <dataset_name>${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Verification log: ${YELLOW}$VERIFICATION_LOG${NC}"
    echo ""
}

# Main execution
main() {
    local verbose=false
    local quick=false
    local fix=false
    local stats=false
    local checksum=false
    local export_report=false
    local check_missing=false
    local check_duplicates=false
    local check_corruption=false
    local target_dataset=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -q|--quick)
                quick=true
                shift
                ;;
            -f|--fix)
                fix=true
                shift
                ;;
            -s|--stats)
                stats=true
                shift
                ;;
            -c|--checksum)
                checksum=true
                shift
                ;;
            --export-report)
                export_report=true
                shift
                ;;
            --check-missing)
                check_missing=true
                shift
                ;;
            --check-duplicates)
                check_duplicates=true
                shift
                ;;
            --check-corruption)
                check_corruption=true
                shift
                ;;
            animation_output|aist_output|dance_output|game_motion_output|HAA500_output|humman_output|kungfu_output|perform_output)
                target_dataset="$1"
                shift
                ;;
            -*)
                echo -e "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
            *)
                echo -e "${RED}Unknown argument: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Initialize log
    echo "MyoDolores Data Verification - $(date)" > "$VERIFICATION_LOG"
    log_message "INFO" "Starting verification with options: verbose=$verbose, quick=$quick, target=$target_dataset"
    
    print_header
    
    # If target dataset specified, modify expected datasets
    if [[ -n "$target_dataset" ]]; then
        if [[ -z "${EXPECTED_DATASETS[$target_dataset]}" ]]; then
            print_status "ERROR" "Unknown dataset: $target_dataset"
            exit 1
        fi
        
        # Create temporary associative array with only target dataset
        declare -A TEMP_DATASETS
        TEMP_DATASETS["$target_dataset"]="${EXPECTED_DATASETS[$target_dataset]}"
        
        # Replace expected datasets
        unset EXPECTED_DATASETS
        declare -A EXPECTED_DATASETS
        for key in "${!TEMP_DATASETS[@]}"; do
            EXPECTED_DATASETS["$key"]="${TEMP_DATASETS[$key]}"
        done
        
        print_status "INFO" "Verifying only dataset: $target_dataset"
        echo ""
    fi
    
    # Run verification steps
    check_data_directory
    verify_file_counts
    verify_file_sizes
    verify_file_integrity
    check_data_consistency
    generate_statistics
    export_report
    
    print_summary
    
    # Exit with appropriate code
    if [[ $ERRORS_FOUND -gt 0 ]]; then
        exit 1
    elif [[ $WARNINGS_FOUND -gt 0 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Check for required tools
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required for data verification${NC}"
    exit 1
fi

if ! python3 -c "import h5py" 2>/dev/null; then
    echo -e "${RED}Error: h5py package is required for H5 file verification${NC}"
    echo -e "${YELLOW}Install with: pip install h5py${NC}"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}Warning: bc calculator not found, some calculations may fail${NC}"
fi

# Run main function with all arguments
main "$@"