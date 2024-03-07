import matplotlib.pyplot as plt
import datasets

class FloorPlanImages:
    def __init__(self, ds_img) -> None:
        self.ds_img = ds_img

        self.id_to_index = {}

        for i, id in enumerate(self.ds_img["id"]):
            self.id_to_index[id] = i
    
    def __getitem__(self, id):
        return datasets.Image(decode=True).decode_example(self.ds_img[self.id_to_index[id]]["img"])
    

class VisualizeRetrievals:

    def __init__(self, images: FloorPlanImages):
        self.images = images

    def visualize_query_by_id(self, query_id, retrieved_ids, titles=None, relevants=None):
        """Visualize the search results.
        
        First item of the lists is the query."""

        k = len(retrieved_ids) + 1

        # if axes is None:
        fig, axes = plt.subplots(1, k, dpi=150, figsize=(20 * k / 5 * 0.75, 7.5 * 0.75))
        fig.tight_layout(pad=1.0)

        axes[0].imshow(self.images[query_id])

        axes[0].set_title(f"{query_id=}")
        axes[0].axis("off")

        for i, id in enumerate(retrieved_ids):

            axes[i+1].imshow(self.images[id])

            if titles is None:
                axes[i + 1].set_title(f"{id=}")
            else:
                axes[i + 1].set_title(f"{id=}\n{titles[i]}")

            axes[i + 1].axis("off")

        return fig


    def visualize_query_by_image(self, retrieved_ids, query_img, titles=None, query_img_title="query"):
        """Visualize the search results.
        
        First item of the lists is the query."""

        k = len(retrieved_ids) + 1

        # if axes is None:
        fig, axes = plt.subplots(1, k, dpi=150, figsize=(20 * k / 5 * 0.75, 7.5 * 0.75))
        fig.tight_layout(pad=1.0)

        axes[0].axis("off")
        axes[0].imshow(query_img)
        axes[0].set_title(query_img_title)

        for i, id in enumerate(retrieved_ids):

            img = self.images[id]            

            axes[i+1].imshow(img)

            if titles is None:
                axes[i + 1].set_title(f"{id=}")
            else:
                axes[i + 1].set_title(f"{id=}\n{titles[i]}")

            axes[i + 1].axis("off")

        return fig
