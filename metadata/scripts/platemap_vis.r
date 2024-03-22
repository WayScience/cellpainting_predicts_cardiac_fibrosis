suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(platetools))

platemap_files <- list.files(pattern = "_platemap\\.csv$", full.names = TRUE)
print(platemap_files)

output_fig_dir <- file.path("platemap_figures")
platemap_suffix <- "_platemap_figure.png"

# Define output figure paths
output_platemap_files <- list()
for (platemap_file in platemap_files) {
    # Extract plate name and remove suffix 
    plate <- basename(platemap_file)
    plate <- stringr::str_remove(plate, "_platemap.csv") 
    
    output_platemap_files[[plate]] <- file.path(output_fig_dir, paste0(plate, platemap_suffix))
}

print(output_platemap_files)


# Load in all platemap CSV files
platemap_dfs <- list()
for (plate in names(output_platemap_files)) {
    # Find the umap file associated with the plate
    platemap_file <- platemap_files[stringr::str_detect(platemap_files, plate)]
    
    # Load in the umap data
    df <- readr::read_csv(
    platemap_file,
    col_types = readr::cols(.default = "c")
)

    platemap_dfs[[plate]] <- df 
}

print(platemap_dfs)

for (plate in names(platemap_dfs)) {
    if (plate %in% c("localhost220512140003_KK22-05-198", "localhost220513100001_KK22-05-198_FactinAdjusted")) {
    # output for each plate
    output_file <- output_platemap_files[[plate]]
    output_file <- paste0(output_file)
    
    platemap <-
        platetools::raw_map(
            data = log10(as.numeric(platemap_dfs[[plate]]$dose)),
            well = platemap_dfs[[plate]]$well_position,
            plate = 96,
            size = 8
        ) +
        ggtitle(paste("Platemap layout for plate", plate)) +
        theme(plot.title = element_text(size = 10, face = "bold")) +
        ggplot2::geom_point(aes(shape = platemap_dfs[[plate]]$treatment)) +
        ggplot2::scale_shape_discrete(name = "Treatment") +
        ggplot2::scale_fill_gradient2(
        name = "log10 of dose (uM)",
        low = "white",
        mid = "pink",
        high = "red",
        )   

    ggsave(
        output_file,
        platemap,
        dpi = 500,
        height = 3.5,
        width = 6
    )
    }
}

for (plate in names(platemap_dfs)) {
    if (plate %in% c("localhost230405150001")) {
    # output for each plate
    output_file <- output_platemap_files[[plate]]
    output_file <- paste0(output_file)
    
    platemap <-
        platetools::raw_map(
            data = platemap_dfs[[plate]]$treatment,
            well = platemap_dfs[[plate]]$well_position,
            plate = 96,
            size = 8
        ) +
        ggtitle(paste("Platemap layout for plate", plate)) +
        theme(plot.title = element_text(hjust=0.5, size = 10, face = "bold", margin = margin(b = -5))) +
        ggplot2::geom_point(aes(shape = platemap_dfs[[plate]]$cell_type)) +
        ggplot2::scale_shape_discrete(name = "Cell Type") +
        ggplot2::scale_fill_discrete(name = "Treatment")

    ggsave(
        output_file,
        platemap,
        dpi = 500,
        height = 3.5,
        width = 6
    )
    }
}

for (plate in names(platemap_dfs)) {
    if (plate %in% c("localhost231120090001")) {
    # output for each plate
    output_file <- output_platemap_files[[plate]]
    output_file <- paste0(output_file)
    
    platemap <-
        platetools::raw_map(
            data = platemap_dfs[[plate]]$heart_number,
            well = platemap_dfs[[plate]]$well_position,
            plate = 96,
            size = 8
        ) +
        ggtitle(paste("Platemap layout for plate", plate)) +
        theme(plot.title = element_text(hjust=0.5, size = 10, face = "bold", margin = margin(b = -5))) +
        ggplot2::geom_point(aes(shape = platemap_dfs[[plate]]$cell_type)) +
        ggplot2::scale_shape_discrete(name = "Cell Type") +
        ggplot2::scale_fill_discrete(name = "Heart Number") +
        theme(
            legend.position = "right",
            legend.title = element_text(size = 9),
            legend.text = element_text(size = 8),
        ) +
        guides(
        shape = guide_legend(override.aes = list(size = 2)),
        fill = guide_legend(override.aes = list(size = 5))
    )

    ggsave(
        output_file,
        platemap,
        dpi = 500,
        height = 3.5,
        width = 6
    )
    }
}

for (plate in names(platemap_dfs)) {
    if (plate %in% c("localhost240201110001")) {
    # output for each plate
    output_file <- output_platemap_files[[plate]]
    output_file <- paste0(output_file)
    
    platemap <-
        platetools::raw_map(
            data = platemap_dfs[[plate]]$treatment,
            well = platemap_dfs[[plate]]$well_position,
            plate = 96,
            size = 8
        ) +
        ggtitle(paste("Platemap layout for plate", plate)) +
        theme(plot.title = element_text(hjust=0.5, size = 10, face = "bold", margin = margin(b = -5))) +
        ggplot2::scale_fill_discrete(name = "Treatment")

    ggsave(
        output_file,
        platemap,
        dpi = 500,
        height = 3.5,
        width = 6
    )
    }
}
