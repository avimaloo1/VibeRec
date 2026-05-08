class FeedbackLoop:
    def __init__(self):
        self.disliked = set()
        self.liked = set()

    def like(self, track_id):
        self.liked.add(track_id)
        if track_id in self.disliked:
            self.disliked.remove(track_id)

    def dislike(self, track_id):
        self.disliked.add(track_id)
        if track_id in self.liked:
            self.liked.remove(track_id)

    def filter_recommendations(self, recommendations):
        """
        Remove disliked songs from recommendation list
        """
        return [
            rec for rec in recommendations
            if rec['track_id'] not in self.disliked
        ]