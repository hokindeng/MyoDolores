#!/bin/bash

# MyoData S3 Sync Script
# Syncs motion capture datasets from S3 to local myo_data directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
S3_BUCKET="myo-data"
LOCAL_DATA_DIR="myo_data"
AWS_PROFILE="${AWS_PROFILE:-default}"

# Dataset paths
DATASETS=(
    "animation_output"
    "aist_output"
    "dance_output"
    "game_motion_output"
    "HAA500_output"
    "humman_output"
    "kungfu_output"
    "perform_output"
)

# Functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}           MyoData S3 Sync Tool${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -l, --list      List available datasets"
    echo "  -a, --all       Sync all datasets (default)"
    echo "  -d, --dry-run   Show what would be synced without downloading"
    echo "  -f, --force     Force re-download even if files exist"
    echo "  --profile PROF  AWS profile to use (default: $AWS_PROFILE)"
    echo ""
    echo "Available datasets:"
    for dataset in "${DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Sync all datasets"
    echo "  $0 HAA500_output           # Sync only HAA500 dataset"
    echo "  $0 --dry-run --all         # Preview all sync operations"
    echo "  $0 --profile my-aws-profile HAA500_output"
}

check_requirements() {
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI not installed${NC}"
        echo "Please install AWS CLI: https://aws.amazon.com/cli/"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        echo -e "${RED}Error: AWS credentials not configured for profile '$AWS_PROFILE'${NC}"
        echo "Please run: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
}

check_s3_access() {
    local dataset=$1
    echo -e "${YELLOW}Checking S3 access for s3://$S3_BUCKET/$dataset/${NC}"
    
    if ! aws s3 ls "s3://$S3_BUCKET/$dataset/" --profile "$AWS_PROFILE" &> /dev/null; then
        echo -e "${RED}Error: Cannot access s3://$S3_BUCKET/$dataset/${NC}"
        echo "Please check your AWS permissions and bucket access"
        return 1
    fi
    return 0
}

sync_dataset() {
    local dataset=$1
    local dry_run=$2
    local force=$3
    
    local s3_path="s3://$S3_BUCKET/$dataset/"
    local local_path="$LOCAL_DATA_DIR/$dataset/"
    
    echo -e "${BLUE}Syncing dataset: $dataset${NC}"
    echo -e "Source: $s3_path"
    echo -e "Target: $local_path"
    
    # Check if dataset exists in S3
    if ! check_s3_access "$dataset"; then
        return 1
    fi
    
    # Create local directory
    mkdir -p "$local_path"
    
    # Prepare sync command
    local sync_cmd="aws s3 sync \"$s3_path\" \"$local_path\" --profile \"$AWS_PROFILE\""
    
    if [[ "$dry_run" == "true" ]]; then
        sync_cmd="$sync_cmd --dryrun"
    fi
    
    if [[ "$force" == "true" ]]; then
        sync_cmd="$sync_cmd --delete"
    fi
    
    echo -e "${YELLOW}Running: $sync_cmd${NC}"
    
    # Execute sync
    if eval "$sync_cmd"; then
        if [[ "$dry_run" != "true" ]]; then
            echo -e "${GREEN}✓ Successfully synced $dataset${NC}"
            
            # Get size info
            local size=$(du -sh "$local_path" 2>/dev/null | cut -f1 || echo "Unknown")
            local files=$(find "$local_path" -type f | wc -l 2>/dev/null || echo "Unknown")
            echo -e "  Size: $size, Files: $files"
        fi
    else
        echo -e "${RED}✗ Failed to sync $dataset${NC}"
        return 1
    fi
    
    echo ""
}

list_datasets() {
    echo -e "${BLUE}Available datasets in S3:${NC}"
    echo ""
    
    for dataset in "${DATASETS[@]}"; do
        echo -n "  $dataset: "
        
        # Check if dataset exists and get basic info
        if aws s3 ls "s3://$S3_BUCKET/$dataset/" --profile "$AWS_PROFILE" &> /dev/null; then
            local objects=$(aws s3 ls "s3://$S3_BUCKET/$dataset/" --recursive --profile "$AWS_PROFILE" | wc -l)
            echo -e "${GREEN}$objects objects${NC}"
        else
            echo -e "${RED}Not accessible${NC}"
        fi
    done
    echo ""
}

main() {
    local sync_all=false
    local dry_run=false
    local force=false
    local target_dataset=""
    local show_list=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -l|--list)
                show_list=true
                shift
                ;;
            -a|--all)
                sync_all=true
                shift
                ;;
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            --profile)
                AWS_PROFILE="$2"
                shift 2
                ;;
            -*)
                echo -e "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
            *)
                target_dataset="$1"
                shift
                ;;
        esac
    done
    
    print_header
    
    # Check requirements
    check_requirements
    
    # Handle list command
    if [[ "$show_list" == "true" ]]; then
        list_datasets
        exit 0
    fi
    
    # Default to sync all if no specific dataset provided
    if [[ -z "$target_dataset" ]]; then
        sync_all=true
    fi
    
    # Validate target dataset if provided
    if [[ -n "$target_dataset" ]]; then
        local valid=false
        for dataset in "${DATASETS[@]}"; do
            if [[ "$dataset" == "$target_dataset" ]]; then
                valid=true
                break
            fi
        done
        
        if [[ "$valid" == "false" ]]; then
            echo -e "${RED}Error: Invalid dataset '$target_dataset'${NC}"
            echo "Use --list to see available datasets"
            exit 1
        fi
    fi
    
    # Sync datasets
    local failed_count=0
    
    if [[ "$sync_all" == "true" ]]; then
        echo -e "${YELLOW}Syncing all datasets...${NC}"
        echo ""
        
        for dataset in "${DATASETS[@]}"; do
            if ! sync_dataset "$dataset" "$dry_run" "$force"; then
                ((failed_count++))
            fi
        done
    else
        sync_dataset "$target_dataset" "$dry_run" "$force" || ((failed_count++))
    fi
    
    # Summary
    echo -e "${BLUE}================================================${NC}"
    if [[ "$failed_count" -eq 0 ]]; then
        echo -e "${GREEN}✓ All sync operations completed successfully${NC}"
    else
        echo -e "${RED}✗ $failed_count sync operations failed${NC}"
        exit 1
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        echo -e "${YELLOW}Note: This was a dry run. No files were actually downloaded.${NC}"
    fi
}

# Run main function with all arguments
main "$@"